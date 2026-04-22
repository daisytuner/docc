#include "sdfg/codegen/code_generators/cuda_code_generator.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/helpers/helpers.h"

namespace sdfg {
namespace codegen {

CUDACodeGenerator::CUDACodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan,
    std::shared_ptr<CodeSnippetFactory> library_snippet_factory
)
    : CodeGenerator(sdfg, analysis_manager, instrumentation_plan, arg_capture_plan, std::move(library_snippet_factory)),
      language_extension_(sdfg) {
    if (sdfg.type() != FunctionType_NV_GLOBAL) {
        throw std::runtime_error("CUDACodeGenerator can only be used for GPU SDFGs");
    }
    if (!arg_capture_plan_.is_empty()) {
        DEBUG_PRINTLN("CUDACodeGenerator does not support capturing args/results!");
    }
};

bool CUDACodeGenerator::generate() {
    this->dispatch_includes();
    this->dispatch_structures();
    this->dispatch_globals();
    this->dispatch_schedule();
    return true;
};

std::string CUDACodeGenerator::function_definition() {
    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : sdfg_.arguments()) {
        args.push_back(language_extension_.declaration(container, sdfg_.type(container)));
    }
    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");

    return "extern \"C\" __global__ void " + sdfg_.name() + "(" + arglist.str() + ")";
};

bool CUDACodeGenerator::emit_header(PrettyPrinter& out) {
    out << "#pragma once" << std::endl;
    dispatch_header_includes(out);
    dispatch_header_structures(out);
    return true;
}

bool CUDACodeGenerator::emit_main_source(std::ostream& out, const std::filesystem::path& header_path) {
    out << "#include \"" << header_path.filename().string() << "\"" << std::endl;
    dispatch_globals();
    dispatch_schedule();

    out << this->globals_stream_.str() << std::endl;

    append_function_source(out);

    return true;
}

bool CUDACodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_header(header_path, std::ofstream::out);
    if (!ofs_header.is_open()) {
        return false;
    }

    std::ofstream ofs_source(source_path, std::ofstream::out);
    if (!ofs_source.is_open()) {
        return false;
    }

    ofs_header << "#pragma once" << std::endl;
    ofs_header << this->includes_stream_.str() << std::endl;
    ofs_header << this->classes_stream_.str() << std::endl;
    ofs_header.close();

    ofs_source << "#include \"" << header_path.filename().string() << "\"" << std::endl;

    ofs_source << this->globals_stream_.str() << std::endl;

    append_function_source(ofs_source);

    ofs_source.close();

    return true;
};

void CUDACodeGenerator::append_function_source(std::ostream& ofs_source) {
    ofs_source << this->function_definition() << std::endl;
    ofs_source << "{" << std::endl;
    ofs_source << this->main_stream_.str() << std::endl;
    ofs_source << "}" << std::endl;
}

void CUDACodeGenerator::dispatch_includes() { this->dispatch_header_includes(this->includes_stream_); }

void CUDACodeGenerator::dispatch_header_includes(PrettyPrinter& out) {
    out << "#define "
        << "__DAISY_NVVM__" << std::endl;
    out << "#include <cstdint>" << std::endl;
    out << "#include <daisy_rtl/daisy_rtl.h>" << std::endl;
};

void CUDACodeGenerator::dispatch_structures() { this->dispatch_header_structures(this->classes_stream_); }

void CUDACodeGenerator::dispatch_header_structures(PrettyPrinter& out) {
    // Forward declarations
    for (auto& structure : sdfg_.structures()) {
        out << "struct " << structure << ";" << std::endl;
    }

    // Generate topology-sorted structure definitions
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> structures_graph;
    typedef boost::graph_traits<structures_graph>::vertex_descriptor Vertex;
    std::vector<std::string> names;
    for (auto& structure : sdfg_.structures()) {
        names.push_back(structure);
    }
    structures_graph graph(names.size());

    for (auto& structure : names) {
        auto& definition = sdfg_.structure(structure);
        for (size_t i = 0; i < definition.num_members(); i++) {
            auto member_type = &definition.member_type(symbolic::integer(i));
            while (dynamic_cast<const types::Array*>(member_type)) {
                auto array_type = static_cast<const types::Array*>(member_type);
                member_type = &array_type->element_type();
            }

            if (auto member_structure = dynamic_cast<const sdfg::types::Structure*>(member_type)) {
                boost::add_edge(
                    std::find(names.begin(), names.end(), member_structure->name()) - names.begin(),
                    std::find(names.begin(), names.end(), structure) - names.begin(),
                    graph
                );
            }
        }
    }

    std::list<Vertex> order;
    std::unordered_map<Vertex, boost::default_color_type> vertex_colors;
    boost::topological_sort(
        graph, std::back_inserter(order), boost::color_map(boost::make_assoc_property_map(vertex_colors))
    );
    order.reverse();

    for (auto& structure_index : order) {
        std::string structure = names.at(structure_index);
        auto& definition = sdfg_.structure(structure);
        out << "struct ";
        if (definition.is_packed()) {
            out << "__attribute__((packed)) ";
        }
        out << structure << std::endl;
        out << "{\n";

        for (size_t i = 0; i < definition.num_members(); i++) {
            auto& member_type = definition.member_type(symbolic::integer(i));
            if (dynamic_cast<const sdfg::types::Structure*>(&member_type)) {
                out << "struct ";
            }
            out << language_extension_.declaration("member_" + std::to_string(i), member_type, false, true);
            out << ";" << std::endl;
        }

        out << "};" << std::endl;
    }
};

void CUDACodeGenerator::dispatch_globals() {
    for (auto& container : sdfg_.externals()) {
        auto& type = sdfg_.type(container);
        if (type.storage_type().is_nv_global()) {
            auto& base_type = dynamic_cast<const types::Pointer&>(type).pointee_type();
            if (sdfg_.linkage_type(container) == LinkageType_External) {
                this->globals_stream_ << "extern " << language_extension_.declaration(container, base_type) << ";"
                                      << std::endl;
            } else {
                this->globals_stream_ << "static " << language_extension_.declaration(container, base_type);
                if (!type.initializer().empty()) {
                    this->globals_stream_ << " = " << type.initializer();
                }
                this->globals_stream_ << ";" << std::endl;
            }
        }
        if (type.storage_type().is_nv_constant()) {
            assert(type.initializer().empty());
            auto& base_type = dynamic_cast<const types::Pointer&>(type).pointee_type();
            this->globals_stream_ << "__constant__ " << language_extension_.declaration(container, base_type, true)
                                  << ";" << std::endl;
        }
    }
};

void CUDACodeGenerator::dispatch_schedule() {
    // Declare shared memory
    for (auto& container : sdfg_.externals()) {
        auto& type = sdfg_.type(container);
        if (type.storage_type().is_nv_shared()) {
            this->main_stream_ << language_extension_.declaration(container, sdfg_.type(container)) << ";" << std::endl;
        }
    }

    // Declare transient containers
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_transient(container)) {
            continue;
        }

        std::string val = this->language_extension_.declaration(container, sdfg_.type(container), false, true);
        if (!val.empty()) {
            this->main_stream_ << val;
            this->main_stream_ << ";" << std::endl;
        }
    }

    auto dispatcher = create_dispatcher(
        language_extension_, sdfg_, analysis_manager_, sdfg_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, *this->library_snippet_factory_);
};

} // namespace codegen
} // namespace sdfg
