#include "docc/target/axelera/axscript_code_generator.h"

#include <fstream>
#include <ranges>
#include <unordered_set>

#include <sdfg/analysis/arguments_analysis.h>
#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/structured_control_flow/return.h>

#include "docc/target/axelera/axelera_scheduler.h"
#include "docc/target/axelera/datapath.h"
#include "docc/target/axelera/schedule.h"

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
        }
        datapaths_.push_back(loop);
    }

    for (size_t i = 0; i < root.size(); ++i) {
        auto& child = root.at(i);
        if (fused.contains(i) || dynamic_cast<structured_control_flow::StructuredLoop*>(&child)) {
            continue;
        }
        if (auto block = dynamic_cast<structured_control_flow::Block*>(&child);
            block && std::ranges::empty(block->dataflow().nodes())) {
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
    auto& sizes = arguments_analysis.argument_sizes(analysis_manager_, sdfg_.root(), false);
    auto& element_sizes = arguments_analysis.argument_element_sizes(analysis_manager_, sdfg_.root(), false);

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

        auto elements = symbolic::expand(symbolic::div(sizes.at(name), element_sizes.at(name)));
        if (!SymEngine::is_a<SymEngine::Integer>(*elements)) {
            unsupported("argument " + name + " of dynamic size " + elements->__str__());
        }
        auto count = SymEngine::down_cast<const SymEngine::Integer&>(*elements).as_int();
        language_extension_.vector_type(element->primitive_type());
        buffers_.push_back({name, count, written});
    }
}

int64_t AxScriptCodeGenerator::Buffer::pwords() const {
    auto lanes = static_cast<int64_t>(PWORD_LANES);
    return (elements + lanes - 1) / lanes;
}

std::string AxScriptCodeGenerator::buffer_type(const Buffer& buffer) const {
    auto& pointer = static_cast<const types::Pointer&>(sdfg_.type(buffer.name));
    auto& element = static_cast<const types::Scalar&>(pointer.pointee_type());
    return "(" + std::to_string(buffer.pwords()) + ",), \"" +
           language_extension_.vector_type(element.primitive_type()) + "\", scope=A.DDR";
}

bool AxScriptCodeGenerator::generate() {
    collect_datapaths();
    collect_buffers();

    this->main_stream_.changeIndent(+8);
    std::vector<std::string> outputs;
    for (auto& buffer : buffers_) {
        if (buffer.is_output) {
            this->main_stream_ << buffer.name << " = A.alloc_buffer(" << buffer_type(buffer) << ")" << std::endl;
            outputs.push_back(buffer.name);
        }
    }
    if (outputs.empty()) {
        unsupported("a kernel without outputs");
    }

    for (auto* datapath : datapaths_) {
        auto dispatcher = codegen::create_dispatcher(
            language_extension_, sdfg_, analysis_manager_, *datapath, instrumentation_plan_, arg_capture_plan_
        );
        dispatcher->dispatch(this->main_stream_, this->globals_stream_, *this->library_snippet_factory_);
    }

    this->main_stream_ << "A.output(" << join(outputs) << ")" << std::endl;
    this->main_stream_.changeIndent(-8);
    return true;
}

std::string AxScriptCodeGenerator::function_definition() {
    std::vector<std::string> parameters;
    for (auto& buffer : buffers_) {
        if (!buffer.is_output) {
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
        arguments.push_back(
            "(\"" + buffer.name + "\", \"" + (buffer.is_output ? "out" : "in") + "\", " +
            std::to_string(buffer.elements) + ")"
        );
    }
    out << "import axelera.compiler.alto.script as A" << std::endl << std::endl;
    out << "# docc: SDFG arguments in order with their element counts (\"out\" buffers are returned through "
           "A.output) and AI cores used."
        << std::endl;
    out << "_docc_arguments = [" << join(arguments) << "]" << std::endl;
    out << "_docc_num_cores = " << num_cores_ << std::endl;
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
