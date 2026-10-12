#include "sdfg/codegen/code_generators/python_code_generator.h"

#include <fstream>

#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"

namespace sdfg {
namespace codegen {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("Python codegen: " + what + " is not supported");
}

std::string join(const std::vector<std::string>& parts) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        result += (i > 0 ? ", " : "") + parts[i];
    }
    return result;
}

/// Innermost scalar type and the extents of the nested arrays above it.
const types::Scalar& unwrap_arrays(const types::IType& type, std::vector<symbolic::Expression>& dims) {
    const types::IType* current = &type;
    while (auto array = dynamic_cast<const types::Array*>(current)) {
        dims.push_back(array->num_elements());
        current = &array->element_type();
    }
    auto scalar = dynamic_cast<const types::Scalar*>(current);
    if (!scalar) {
        unsupported("containers of " + type.print());
    }
    return *scalar;
}

} // namespace

PythonCodeGenerator::PythonCodeGenerator(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    InstrumentationPlan& instrumentation_plan,
    ArgCapturePlan& arg_capture_plan,
    std::shared_ptr<CodeSnippetFactory> library_snippet_factory,
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
    if (sdfg.type() != FunctionType_CPU) {
        unsupported("non-CPU SDFGs");
    }
    if (!instrumentation_plan.is_empty() || !arg_capture_plan.is_empty()) {
        unsupported("instrumentation and argument capture");
    }
}

void PythonCodeGenerator::validate(structured_control_flow::ControlFlowNode& node) const {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto& dataflow_node : block->dataflow().nodes()) {
            if (auto libnode = dynamic_cast<const data_flow::LibraryNode*>(&dataflow_node)) {
                auto id = libnode->code().value() + "::" + ImplementationType_Python.value();
                if (!LibraryNodeDispatcherRegistry::instance().get_library_node_dispatcher(id)) {
                    unsupported("library node " + libnode->code().value() + " (expand it before code generation)");
                }
            }
        }
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            validate(sequence->at(i));
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            validate(if_else->at(i).first);
        }
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        bool parallel = dynamic_cast<structured_control_flow::Map*>(loop) ||
                        dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (parallel && loop->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
            unsupported("schedule " + loop->schedule_type().value());
        }
        validate(loop->root());
    } else if (auto while_loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        validate(while_loop->root());
    }
}

bool PythonCodeGenerator::generate() {
    validate(sdfg_.root());

    this->includes_stream_ << PythonLanguageExtension::runtime_prelude();

    this->main_stream_.changeIndent(+4);
    for (auto& container : sdfg_.arguments()) {
        auto pointer = dynamic_cast<const types::Pointer*>(&sdfg_.type(container));
        if (!pointer || !pointer->has_pointee_type() || !dynamic_cast<const types::Array*>(&pointer->pointee_type())) {
            continue;
        }
        // Callers pass flat views; pointers to fixed-size arrays index them as rows.
        std::vector<symbolic::Expression> dims;
        unwrap_arrays(pointer->pointee_type(), dims);
        std::vector<std::string> extents;
        for (auto& dim : dims) {
            extents.push_back(language_extension_.expression(dim));
        }
        this->main_stream_ << container << " = " << container << ".reshape(-1, " << join(extents) << ")" << std::endl;
    }

    this->main_stream_ << "with np.errstate(all=\"ignore\"):" << std::endl;
    this->main_stream_.changeIndent(+4);
    auto mark = this->main_stream_.content();

    dispatch_locals();
    auto dispatcher = create_dispatcher(
        language_extension_, sdfg_, analysis_manager_, sdfg_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher->dispatch(this->main_stream_, this->globals_stream_, *this->library_snippet_factory_);

    if (this->main_stream_.content() == mark) {
        this->main_stream_ << "pass" << std::endl;
    }
    this->main_stream_.changeIndent(-8);
    return true;
}

void PythonCodeGenerator::dispatch_locals() {
    for (auto& container : sdfg_.containers()) {
        if (!sdfg_.is_transient(container) || sdfg_.is_external(container)) {
            continue;
        }
        auto& type = sdfg_.type(container);

        if (auto scalar = dynamic_cast<const types::Scalar*>(&type)) {
            // C leaves locals uninitialized; Python needs the name bound before a first read.
            this->main_stream_ << container << " = " << language_extension_.zero(scalar->primitive_type()) << std::endl;
        } else if (dynamic_cast<const types::Array*>(&type)) {
            std::vector<symbolic::Expression> dims;
            auto& element = unwrap_arrays(type, dims);
            std::vector<std::string> extents;
            for (auto& dim : dims) {
                extents.push_back(language_extension_.expression(dim));
            }
            this->main_stream_ << container << " = np.zeros((" << join(extents) << ",), "
                               << language_extension_.primitive_type(element.primitive_type()) << ")" << std::endl;
        } else if (dynamic_cast<const types::Pointer*>(&type)) {
            auto storage = type.storage_type();
            if (storage.allocation() != types::StorageType::AllocationType::Managed) {
                this->main_stream_ << container << " = None" << std::endl;
                continue;
            }
            if (!storage.is_cpu_stack() && !storage.is_cpu_heap()) {
                unsupported("storage " + storage.value() + " of " + container);
            }
            this->main_stream_
                << container << " = "
                << language_extension_.allocation(type, language_extension_.expression(storage.allocation_size()))
                << std::endl;
        } else {
            unsupported("containers of " + type.print());
        }
    }
}

std::string PythonCodeGenerator::function_definition() {
    return "def " + this->externals_prefix_ + sdfg_.name() + "(" + join(sdfg_.arguments()) + "):";
}

bool PythonCodeGenerator::emit_header(PrettyPrinter& out) {
    return true;
}

bool PythonCodeGenerator::emit_main_source(std::ostream& out, const std::filesystem::path& header_path) {
    out << this->includes_stream_.str() << std::endl;
    if (!this->globals_stream_.str().empty()) {
        out << this->globals_stream_.str() << std::endl;
    }
    append_function_source(out);
    return true;
}

bool PythonCodeGenerator::as_source(const std::filesystem::path& header_path, const std::filesystem::path& source_path) {
    std::ofstream ofs_source(source_path, std::ofstream::out | std::ofstream::trunc);
    if (!ofs_source.is_open()) {
        return false;
    }
    return emit_main_source(ofs_source, header_path);
}

void PythonCodeGenerator::append_function_source(std::ostream& ofs_source) {
    ofs_source << std::endl << this->function_definition() << std::endl;
    ofs_source << this->main_stream_.str();
}

} // namespace codegen
} // namespace sdfg
