#include "sdfg/targets/cuda/cuda_reduce_dispatcher.h"

#include <memory>
#include <string>
#include <vector>

#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace cuda {

CUDAReduceDispatcher::CUDAReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : gpu::GPUReduceDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan) {
      };

std::string CUDAReduceDispatcher::schedule_value() const { return ScheduleType_CUDA::value(); }

codegen::TargetType CUDAReduceDispatcher::target_type() const { return TargetType_CUDA; }

std::unique_ptr<codegen::LanguageExtension> CUDAReduceDispatcher::create_device_language_extension() const {
    return std::make_unique<codegen::CUDALanguageExtension>(sdfg_);
}

bool CUDAReduceDispatcher::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.is_nv_generic();
}

std::string CUDAReduceDispatcher::kernel_file_extension() const { return "cu"; }

void CUDAReduceDispatcher::emit_kernel_includes(codegen::CodeSnippetFactory& library_snippet_factory) const {
    library_snippet_factory.add_global("#include <cstdio>");
    library_snippet_factory.add_global("#include <math.h>");
}

void CUDAReduceDispatcher::emit_library_preamble(codegen::PrettyPrinter& /*library_stream*/) const {}

void CUDAReduceDispatcher::emit_kernel_call(
    codegen::PrettyPrinter& main_stream,
    const std::string& kernel_name,
    symbolic::Expression& num_blocks,
    symbolic::Expression& block_size,
    std::vector<std::string>& arguments_device
) {
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    main_stream << kernel_name << "<<<";
    main_stream << "dim3((int)(" << this->language_extension_.expression(num_blocks) << "), (int)(1), (int)(1)), ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(block_size) << "), (int)(1), (int)(1))";
    main_stream << ">>>";
    main_stream << "(";
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    check_cuda_kernel_launch_errors(main_stream, this->language_extension_, instrumentation_plan_.should_instrument(node_));

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

} // namespace cuda
} // namespace sdfg
