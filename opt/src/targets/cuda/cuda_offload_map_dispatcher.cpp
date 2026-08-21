#include "sdfg/targets/cuda/cuda_offload_map_dispatcher.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace cuda {

CUDAOffloadMapDispatcher::CUDAOffloadMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : gpu::GPUOffloadMapDispatcher(
          language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
      ),
      kernel_language_extension_(sdfg) {};

codegen::LanguageExtension& CUDAOffloadMapDispatcher::create_kernel_language_extension() {
    return kernel_language_extension_;
}

void CUDAOffloadMapDispatcher::dispatch_kernel_call(
    codegen::PrettyPrinter& main_stream,
    const std::string& kernel_name,
    symbolic::Expression& num_blocks_x,
    symbolic::Expression& num_blocks_y,
    symbolic::Expression& num_blocks_z,
    symbolic::Expression& block_size_x,
    symbolic::Expression& block_size_y,
    symbolic::Expression& block_size_z,
    std::vector<std::string>& arguments_device
) {
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Kernel launch
    main_stream << kernel_name << "<<<";
    main_stream << "dim3((int)(" << this->language_extension_.expression(num_blocks_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_z) << ")), ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(block_size_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_z) << "))";
    main_stream << ">>>(";
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    // Synchronize / check launch errors
    this->dispatch_kernel_launch_error_check(
        main_stream, this->language_extension_, instrumentation_plan_.should_instrument(node_)
    );

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

void CUDAOffloadMapDispatcher::dispatch_kernel_launch_error_check(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
) {
    check_cuda_kernel_launch_errors(stream, language_extension, instrumented);
}

codegen::InstrumentationInfo CUDAOffloadMapDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        std::string flop_str = language_extension_.expression(flop);
        metrics.insert({"flop", flop_str});
    }

    return codegen::InstrumentationInfo(
        node_.element_id(),
        node_.element_type(),
        TargetType_CUDA,
        codegen::InstrumentationEventType::CUDA,
        loop_info,
        metrics
    );
};

int CUDAOffloadMapDispatcher::get_warp_size() const { return CUDA_WARP_SIZE; }

bool CUDAOffloadMapDispatcher::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.is_nv_generic();
}

std::string CUDAOffloadMapDispatcher::kernel_file_extension() const { return "cu"; }

} // namespace cuda
} // namespace sdfg
