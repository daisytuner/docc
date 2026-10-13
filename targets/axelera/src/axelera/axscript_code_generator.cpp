#include "docc/target/axelera/axscript_code_generator.h"

#include <fstream>
#include <optional>
#include <ranges>
#include <unordered_set>

#include <sdfg/analysis/arguments_analysis.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/stdlib/free.h>
#include <sdfg/data_flow/library_nodes/stdlib/malloc.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/structured_control_flow/return.h>
#include <sdfg/types/array.h>
#include <sdfg/types/utils.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/axscript_program.h"
#include "docc/target/axelera/datapath.h"
#include "docc/target/axelera/schedule.h"
#include "docc/target/axelera/tile_target.h"

namespace sdfg {
namespace axelera {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("AxScript codegen: " + what + " is not supported");
}

std::string join(const std::vector<std::string>& parts) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        result += (i > 0 ? ", " : "") + parts[i];
    }
    return result;
}

/// True if @p block only allocates or frees transients, which the module allocates as buffers itself.
bool manages_transients(const StructuredSDFG& sdfg, structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    if (dataflow.library_nodes().empty()) {
        return false;
    }
    for (auto& node : dataflow.nodes()) {
        if (dynamic_cast<const stdlib::MallocNode*>(&node) || dynamic_cast<const stdlib::FreeNode*>(&node)) {
            continue;
        }
        auto access = dynamic_cast<const data_flow::AccessNode*>(&node);
        if (!access || dynamic_cast<const data_flow::ConstantNode*>(access) || !sdfg.is_transient(access->data())) {
            return false;
        }
    }
    return true;
}

/// Elements of @p element type that a top-level malloc into @p name allocates, if one does.
std::optional<int64_t> allocated_elements(StructuredSDFG& sdfg, const std::string& name, const types::Scalar& element) {
    auto& root = sdfg.root();
    for (size_t i = 0; i < root.size(); ++i) {
        auto block = dynamic_cast<structured_control_flow::Block*>(&root.at(i));
        if (!block) {
            continue;
        }
        for (auto* node : block->dataflow().library_nodes()) {
            auto malloc = dynamic_cast<stdlib::MallocNode*>(node);
            if (!malloc) {
                continue;
            }
            for (auto& memlet : block->dataflow().out_edges(*malloc)) {
                if (static_cast<const data_flow::AccessNode&>(memlet.dst()).data() != name) {
                    continue;
                }
                auto elements = symbolic::expand(symbolic::div(malloc->size(), types::get_type_size(element, false)));
                if (SymEngine::is_a<SymEngine::Integer>(*elements)) {
                    return SymEngine::down_cast<const SymEngine::Integer&>(*elements).as_int();
                }
            }
        }
    }
    return std::nullopt;
}

} // namespace

AxScriptCodeGenerator::AxScriptCodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan,
    std::shared_ptr<codegen::CodeSnippetFactory> library_snippet_factory,
    const std::string& externals_prefix
)
    : CodeGenerator(
          sdfg,
          analysis_manager,
          instrumentation_plan,
          arg_capture_plan,
          std::move(library_snippet_factory),
          externals_prefix
      ),
      language_extension_(sdfg, externals_prefix) {
    if (!instrumentation_plan.is_empty() || !arg_capture_plan.is_empty()) {
        unsupported("instrumentation and argument capture");
    }
}

void AxScriptCodeGenerator::collect_datapaths() {
    auto& root = sdfg_.root();
    auto cores = analysis_manager_.options().get(NUM_CORES, int64_t{1});
    std::unordered_set<size_t> fused;
    for (size_t i = 0; i < root.size(); ++i) {
        auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&root.at(i));
        if (!loop) {
            continue;
        }
        auto& schedule = loop->schedule_type();
        if (!ScheduleType_Axelera::is(schedule, AxeleraLevel::Core) &&
            !ScheduleType_Axelera::is(schedule, AxeleraLevel::Stream)) {
            auto reason = datapath_obstacle(sdfg_, analysis_manager_, *loop, cores);
            if (reason.empty()) {
                reason = "it was not scheduled for Axelera";
            }
            throw std::runtime_error(
                "AxScript codegen: the loop over " + loop->indvar()->get_name() + " cannot run on Axelera: " + reason
            );
        }
        if (ScheduleType_Axelera::is(schedule, AxeleraLevel::Core)) {
            num_cores_ = std::max(num_cores_, ScheduleType_Axelera::num_cores(schedule));
        }
        if (auto reduce = dynamic_cast<structured_control_flow::Reduce*>(loop)) {
            auto kernel = match_reduction(sdfg_, analysis_manager_, *reduce);
            fused.insert(kernel.init);
            fused.insert(kernel.writeback);
            if (dynamic_cast<const types::Scalar*>(&sdfg_.type(kernel.output.container))) {
                reduction_results_.insert(kernel.output.container);
            }
        }
        datapaths_.push_back(loop);
    }

    for (size_t i = 0; i < root.size(); ++i) {
        auto& child = root.at(i);
        if (fused.contains(i) || dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
            continue;
        }
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child);
            block && (std::ranges::empty(block->dataflow().nodes()) || manages_transients(sdfg_, *block))) {
            continue;
        }
        if (auto assignments = dynamic_cast<structured_control_flow::AssignmentBlock*>(&child);
            assignments && assignments->assignments().empty()) {
            continue;
        }
        // Outputs leave through A.output, so the function's final void return needs no code.
        if (auto ret = dynamic_cast<structured_control_flow::Return*>(&child);
            ret && ret->data().empty() && i + 1 == root.size()) {
            continue;
        }
        unsupported("code outside a datapath loop (element " + std::to_string(child.element_id()) + ")");
    }
}

void AxScriptCodeGenerator::collect_buffers() {
    auto& arguments_analysis = analysis_manager_.get<analysis::ArgumentsAnalysis>();
    auto& arguments = arguments_analysis.arguments(analysis_manager_, sdfg_.root());
    auto matrices = matrix_arguments(sdfg_);
    // Tile copies address their operands through bare pointers, which the analysis cannot size.
    bool sized = arguments_analysis.argument_size_known(analysis_manager_, sdfg_.root(), false);

    for (auto& name : sdfg_.arguments()) {
        auto& type = sdfg_.type(name);
        auto pointer = dynamic_cast<const types::Pointer*>(&type);
        auto element = pointer && pointer->has_pointee_type()
                           ? dynamic_cast<const types::Scalar*>(&pointer->pointee_type())
                           : nullptr;
        if (!element) {
            unsupported("argument " + name + " of type " + type.print());
        }
        auto it = arguments.find(name);
        if (it == arguments.end()) {
            unsupported("unused argument " + name);
        }
        bool read = it->second.is_explicit_input;
        bool written = it->second.is_output;
        if (read && written) {
            unsupported("argument " + name + ", which is both read and written,");
        }
        language_extension_.vector_type(element->primitive_type());

        if (auto matrix = matrices.find(name); matrix != matrices.end()) {
            auto [row_length, count] = matrix->second;
            buffers_.push_back({name, count, written, row_length, element->primitive_type()});
            continue;
        }
        if (!sized) {
            unsupported("argument " + name + " of unknown size");
        }
        auto& sizes = arguments_analysis.argument_sizes(analysis_manager_, sdfg_.root(), false);
        auto& element_sizes = arguments_analysis.argument_element_sizes(analysis_manager_, sdfg_.root(), false);
        auto elements = symbolic::expand(symbolic::div(sizes.at(name), element_sizes.at(name)));
        if (!SymEngine::is_a<SymEngine::Integer>(*elements)) {
            unsupported("argument " + name + " of dynamic size " + elements->__str__());
        }
        auto count = SymEngine::down_cast<const SymEngine::Integer&>(*elements).as_int();
        buffers_.push_back({name, count, written, 0, element->primitive_type()});
    }

    // Arrays and pointers datapaths and programs hand each other; local buffers of programs are allocated by the
    // program. A pointer takes its extent from the tile copies addressing it as a matrix.
    for (auto& name : sdfg_.containers()) {
        auto& type = sdfg_.type(name);
        if (!sdfg_.is_transient(name) || type.storage_type() == StorageType_L1() ||
            type.storage_type() == StorageType_L2() || type.storage_type() == StorageType_IMC()) {
            continue;
        }
        auto matrix = matrices.find(name);
        int64_t row_length = matrix == matrices.end() ? 0 : matrix->second.row_length;
        const types::Scalar* element = nullptr;
        int64_t count = 0;
        if (auto array = dynamic_cast<const types::Array*>(&type)) {
            element = dynamic_cast<const types::Scalar*>(&array->element_type());
            if (!element || !SymEngine::is_a<SymEngine::Integer>(*array->num_elements())) {
                unsupported("local array " + name + " of type " + type.print());
            }
            count = SymEngine::down_cast<const SymEngine::Integer&>(*array->num_elements()).as_int();
        } else if (auto pointer = dynamic_cast<const types::Pointer*>(&type)) {
            element = pointer->has_pointee_type() ? dynamic_cast<const types::Scalar*>(&pointer->pointee_type())
                                                  : nullptr;
            std::optional<int64_t> allocated;
            if (element && matrix == matrices.end()) {
                allocated = allocated_elements(sdfg_, name, *element);
            }
            if (!element || (matrix == matrices.end() && !allocated)) {
                unsupported("local pointer " + name + ", whose extent neither a tile copy nor a malloc determines,");
            }
            count = allocated ? *allocated : matrix->second.elements;
        } else {
            continue;
        }
        language_extension_.vector_type(element->primitive_type());
        buffers_.push_back({name, count, false, row_length, element->primitive_type(), true});
    }
    for (auto& name : reduction_results_) {
        buffers_.push_back({name, 1, false, 0, types::PrimitiveType::Float, true});
        language_extension_.add_uniform(name);
    }

    for (auto& buffer : buffers_) {
        if (buffer.row_length == 0) {
            continue;
        }
        if (buffer.row_length % static_cast<int64_t>(PWORD_LANES) != 0 || buffer.elements % buffer.row_length != 0) {
            unsupported("buffer " + buffer.name + " with rows of " + std::to_string(buffer.row_length) + " elements");
        }
        language_extension_.set_matrix(buffer.name, buffer.row_length / static_cast<int64_t>(PWORD_LANES));
    }
}

int64_t AxScriptCodeGenerator::Buffer::pwords() const {
    auto lanes = static_cast<int64_t>(PWORD_LANES);
    return (elements + lanes - 1) / lanes;
}

std::string AxScriptCodeGenerator::Buffer::shape() const {
    if (row_length == 0) {
        return "(" + std::to_string(pwords()) + ",)";
    }
    return "(" + std::to_string(elements / row_length) + ", " +
           std::to_string(row_length / static_cast<int64_t>(PWORD_LANES)) + ")";
}

std::string AxScriptCodeGenerator::buffer_type(const Buffer& buffer) const {
    return buffer.shape() + ", \"" + language_extension_.vector_type(buffer.element) + "\", scope=A.DDR";
}

bool AxScriptCodeGenerator::generate() {
    collect_datapaths();
    collect_buffers();

    this->main_stream_.changeIndent(+8);
    std::vector<std::string> outputs;
    for (auto& buffer : buffers_) {
        if (buffer.is_output) {
            outputs.push_back(buffer.name);
        }
        if (buffer.is_output || buffer.transient) {
            this->main_stream_ << buffer.name << " = A.alloc_buffer(" << buffer_type(buffer) << ")" << std::endl;
        }
    }
    if (outputs.empty()) {
        unsupported("a kernel without outputs");
    }

    // Datapaths register the constants they read, which must be allocated before them.
    codegen::PrettyPrinter datapaths(this->main_stream_.indent());
    for (auto* datapath : datapaths_) {
        if (is_program(*datapath)) {
            auto cores = dynamic_cast<structured_control_flow::Map*>(datapath);
            if (!cores) {
                unsupported("explicit memory management outside an AI-core loop");
            }
            std::map<std::string, BufferShape> shapes;
            for (auto& buffer : buffers_) {
                if (buffer.row_length != 0) {
                    shapes[buffer.name] = {buffer.element, buffer.elements / buffer.row_length, buffer.row_length};
                }
            }
            ProgramEmitter(sdfg_, language_extension_, datapaths, std::move(shapes)).emit_cores(*cores);
            uses_operators_ = true;
            continue;
        }
        auto dispatcher = codegen::create_dispatcher(
            language_extension_, sdfg_, analysis_manager_, *datapath, instrumentation_plan_, arg_capture_plan_
        );
        dispatcher->dispatch(datapaths, this->globals_stream_, *this->library_snippet_factory_);
    }
    auto lanes = "\"" + language_extension_.vector_type(types::PrimitiveType::Float) + "\"";
    for (size_t k = 0; k < language_extension_.constants().size(); ++k) {
        auto name = AxScriptLanguageExtension::constant_name(k);
        this->main_stream_ << name << " = A.alloc_constant(" << name << "_lanes, " << lanes << ", scope=A.DDR)"
                           << std::endl;
    }
    auto indent = this->main_stream_.indent();
    this->main_stream_.setIndent(0);
    this->main_stream_ << datapaths.str();
    this->main_stream_.setIndent(indent);

    this->main_stream_ << "A.output(" << join(outputs) << ")" << std::endl;
    this->main_stream_.changeIndent(-8);
    return true;
}

std::string AxScriptCodeGenerator::function_definition() {
    std::vector<std::string> parameters;
    for (auto& buffer : buffers_) {
        if (!buffer.is_output && !buffer.transient) {
            parameters.push_back(buffer.name + ": A.Buffer(" + buffer_type(buffer) + ")");
        }
    }
    return "def main(" + join(parameters) + "):";
}

bool AxScriptCodeGenerator::emit_header(codegen::PrettyPrinter& out) {
    return true;
}

bool AxScriptCodeGenerator::emit_main_source(std::ostream& out, const std::filesystem::path& header_path) {
    std::vector<std::string> arguments;
    for (auto& buffer : buffers_) {
        if (buffer.transient) {
            continue;
        }
        arguments.push_back(
            "(\"" + buffer.name + "\", \"" + (buffer.is_output ? "out" : "in") + "\", " +
            std::to_string(buffer.elements) + ", " + buffer.shape() + ")"
        );
    }
    out << "import axelera.compiler.alto.script as A" << std::endl;
    if (uses_operators_) {
        out << "from axelera.compiler.alto.axol import atomic" << std::endl;
    }
    out << std::endl;
    out << "# docc: SDFG arguments in order with their element counts and PWORD buffer shapes (\"out\" buffers are "
           "returned through A.output) and AI cores used."
        << std::endl;
    out << "_docc_arguments = [" << join(arguments) << "]" << std::endl;
    out << "_docc_num_cores = " << num_cores_ << std::endl;
    if (uses_operators_) {
        out << "# docc: the module manages local memory itself, which alto's tiling injection would redo." << std::endl;
        out << "_docc_tiling_injection = False" << std::endl;
    }
    auto& constants = language_extension_.constants();
    for (size_t k = 0; k < constants.size(); ++k) {
        out << AxScriptLanguageExtension::constant_name(k) << "_lanes = ["
            << AxScriptLanguageExtension::float_literal(constants[k]) << "] * " << PWORD_LANES << std::endl;
    }
    append_function_source(out);
    return true;
}

bool AxScriptCodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_source(source_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_source.is_open()) {
        return false;
    }
    return emit_main_source(ofs_source, header_path);
}

void AxScriptCodeGenerator::append_function_source(std::ostream& ofs_source) {
    ofs_source << std::endl << std::endl << "@A.ir_module" << std::endl;
    ofs_source << "class " << this->externals_prefix_ << sdfg_.name() << ":" << std::endl;
    ofs_source << "    @A.prim_func" << std::endl;
    ofs_source << "    " << this->function_definition() << std::endl;
    ofs_source << this->main_stream_.str();
}

} // namespace axelera
} // namespace sdfg
