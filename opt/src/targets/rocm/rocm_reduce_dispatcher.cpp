#include "sdfg/targets/rocm/rocm_reduce_dispatcher.h"

#include <memory>
#include <string>
#include <vector>

#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/codegen/language_extensions/rocm_language_extension.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {

ROCMReduceDispatcher::ROCMReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : gpu::GPUReduceDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan) {
      };

std::string ROCMReduceDispatcher::schedule_value() const { return ScheduleType_ROCM::value(); }

codegen::TargetType ROCMReduceDispatcher::target_type() const { return TargetType_ROCM; }

std::unique_ptr<codegen::LanguageExtension> ROCMReduceDispatcher::create_device_language_extension() const {
    return std::make_unique<codegen::ROCMLanguageExtension>(sdfg_);
}

bool ROCMReduceDispatcher::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.value() == "AMD_Generic";
}

std::string ROCMReduceDispatcher::kernel_file_extension() const { return "rocm.cpp"; }

void ROCMReduceDispatcher::emit_kernel_includes(codegen::CodeSnippetFactory& library_snippet_factory) const {
    library_snippet_factory.add_global("#include <cstdio>");
    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
}

void ROCMReduceDispatcher::emit_library_preamble(codegen::PrettyPrinter& library_stream) const {
    library_stream << "#include <hip/hip_runtime.h>" << std::endl << std::endl;
}

void ROCMReduceDispatcher::emit_kernel_call(
    codegen::PrettyPrinter& main_stream,
    const std::string& kernel_name,
    symbolic::Expression& num_blocks,
    symbolic::Expression& block_size,
    std::vector<std::string>& arguments_device
) {
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    main_stream << "hipLaunchKernelGGL(" << kernel_name << ", ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(num_blocks) << "), (int)(1), (int)(1)), ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(block_size) << "), (int)(1), (int)(1)), ";
    main_stream << "0, 0, "; // shared memory size and stream
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    check_rocm_kernel_launch_errors(main_stream, this->language_extension_);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

} // namespace rocm
} // namespace sdfg
