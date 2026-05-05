#include "sdfg/codegen/code_generators/cpp_code_generator.h"

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"

namespace sdfg {
namespace codegen {

std::string CPPCodeGenerator::function_definition() {
    /********** Arglist **********/
    std::vector<std::string> args;
    for (auto& container : this->sdfg_.arguments()) {
        args.push_back(language_extension_.declaration(container, this->sdfg_.type(container)));
    }

    std::stringstream arglist;
    arglist << sdfg::helpers::join(args, ", ");
    std::string arglist_str = arglist.str();
    if (arglist_str.empty()) {
        arglist_str = "void";
    }

    return "extern \"C\" " + this->language_extension_.declaration("", sdfg_.return_type()) + this->externals_prefix_ +
           sdfg_.name() + "(" + arglist_str + ")";
};

void CPPCodeGenerator::emit_capture_context_init(std::ostream& ofs_source) const {
    std::string name = sdfg_.name();
    std::string arg_capture_path = sdfg_.metadata().at("arg_capture_path");

    ofs_source << "static __daisy_capture_t* __capture_ctx;" << std::endl;
    ofs_source << "static void __attribute__((constructor(1000))) __capture_ctx_init(void) {" << std::endl;
    ofs_source << "\t__capture_ctx = __daisy_capture_init(\"" << name << "\", \"" << arg_capture_path << "\");"
               << std::endl;
    ofs_source << "}" << std::endl;
    ofs_source << std::endl;
}

void CPPCodeGenerator::dispatch_header_includes(PrettyPrinter& out) {
#if defined(__linux__)
    out << "#include <alloca.h>" << std::endl;
    out << "#include <malloc.h>" << std::endl;
#endif

    out << "#include <cmath>" << std::endl;
    out << "#include <cstdio>" << std::endl;
    out << "#include <cstdlib>" << std::endl;
    out << "#include <cstring>" << std::endl;
    out << "#include <cstdint>" << std::endl;

#if defined(__APPLE__)
    out << "#include <Accelerate/Accelerate.h>" << std::endl;
#else
    out << "#include <cblas.h>" << std::endl;
#endif

    out << "#include <daisy_rtl/daisy_rtl.h>" << std::endl;
}

void CPPCodeGenerator::dispatch_header_structures(PrettyPrinter& out) {
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
}

void CPPCodeGenerator::dispatch_globals() {
    // Declare globals
    for (auto& container : sdfg_.externals()) {
        // Function declarations
        if (dynamic_cast<const types::Function*>(&sdfg_.type(container))) {
            this->globals_stream_ << "extern \"C\" ";
            this->globals_stream_
                << language_extension_.declaration(this->externals_prefix_ + container, sdfg_.type(container)) << ";"
                << std::endl;
            continue;
        }

        // Other types must be pointers
        auto& type = dynamic_cast<const types::Pointer&>(sdfg_.type(container));
        assert(type.has_pointee_type() && "Externals must have a pointee type");
        auto& base_type = type.pointee_type();

        if (sdfg_.linkage_type(container) == LinkageType_External) {
            this->globals_stream_ << "extern "
                                  << language_extension_.declaration(this->externals_prefix_ + container, base_type)
                                  << ";" << std::endl;
        } else {
            this->globals_stream_ << "static "
                                  << language_extension_.declaration(this->externals_prefix_ + container, base_type);
            if (!type.initializer().empty()) {
                this->globals_stream_ << " = " << type.initializer();
            }
            this->globals_stream_ << ";" << std::endl;
        }
    }
};

void CPPCodeGenerator::dispatch_schedule() {
    // Allocate variables
    for (auto& container : sdfg_.containers()) {
        if (sdfg_.is_external(container)) {
            continue;
        }
        auto& type = sdfg_.type(container);

        // Declare transient
        if (sdfg_.is_transient(container)) {
            std::string val = this->language_extension_.declaration(container, sdfg_.type(container), false, true);
            if (!val.empty()) {
                this->main_stream_ << val;
                this->main_stream_ << ";" << std::endl;
            }
        }

        // Allocate if needed
        if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
            assert(
                !type.storage_type().allocation_size().is_null() &&
                "Managed allocations must have a valid allocation size"
            );
            if (type.storage_type().is_cpu_stack()) {
                this->main_stream_ << container << " = ";
                this->main_stream_
                    << "alloca(" << this->language_extension_.expression(type.storage_type().allocation_size()) << ")";
                this->main_stream_ << ";" << std::endl;
            } else if (type.storage_type().is_cpu_heap()) {
                this->main_stream_ << container << " = ";
                this->main_stream_ << this->externals_prefix_ << "malloc("
                                   << this->language_extension_.expression(type.storage_type().allocation_size())
                                   << ")";
                this->main_stream_ << ";" << std::endl;
            } else if (type.storage_type().is_nv_generic()) {
                this->main_stream_ << "cudaSetDevice(0);" << std::endl;
                this->main_stream_ << "cudaMalloc(&" << container << ", "
                                   << this->language_extension_.expression(type.storage_type().allocation_size())
                                   << ");" << std::endl;
            }
        }
    }

    auto dispatcher = create_dispatcher(
        language_extension_, sdfg_, analysis_manager_, sdfg_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, *this->library_snippet_factory_);

    if (sdfg_.root().size() == 0 ||
        !dynamic_cast<const structured_control_flow::Return*>(&(sdfg_.root().at(sdfg_.root().size() - 1)).first)) {
        // Free heap allocations
        for (auto& container : sdfg_.containers()) {
            if (sdfg_.is_external(container)) {
                continue;
            }
            auto& type = sdfg_.type(container);

            // Free if needed
            if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
                if (type.storage_type().is_cpu_heap()) {
                    this->main_stream_ << this->externals_prefix_ << "free(" << container << ");" << std::endl;
                } else if (type.storage_type().is_nv_generic()) {
                    this->main_stream_ << "cudaSetDevice(0);" << std::endl;
                    this->main_stream_ << "cudaFree(" << container << ");" << std::endl;
                }
            }
        }
    }
};

} // namespace codegen
} // namespace sdfg
