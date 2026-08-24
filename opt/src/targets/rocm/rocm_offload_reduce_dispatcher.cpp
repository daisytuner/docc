#include "sdfg/targets/rocm/rocm_offload_reduce_dispatcher.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/flop_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/helpers/helpers.h>

#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {

ROCMOffloadReduceDispatcher::ROCMOffloadReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : gpu::GPUOffloadReduceDispatcher(
          language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan
      ),
      kernel_language_extension_(sdfg) {};

codegen::LanguageExtension& ROCMOffloadReduceDispatcher::create_kernel_language_extension() {
    return kernel_language_extension_;
}

void ROCMOffloadReduceDispatcher::dispatch_kernel_call(
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
    main_stream << "hipLaunchKernelGGL(" << kernel_name << ", ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(num_blocks_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_z) << ")), ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(block_size_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_z) << ")), ";
    main_stream << "0, 0, "; // shared memory size and stream
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

void ROCMOffloadReduceDispatcher::dispatch_kernel_launch_error_check(
    codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool instrumented
) {
    (void) instrumented; // ROCm launch-error check does not take an instrumentation flag
    check_rocm_kernel_launch_errors(stream, language_extension);
}

codegen::InstrumentationInfo ROCMOffloadReduceDispatcher::instrumentation_info() const {
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
        TargetType_ROCM,
        codegen::InstrumentationEventType::CUDA,
        loop_info,
        metrics
    );
};

int ROCMOffloadReduceDispatcher::get_warp_size() const { return rocm_wavefront_size(); }

bool ROCMOffloadReduceDispatcher::is_device_pointer_storage(const types::StorageType& storage) const {
    return storage.is_amd_generic();
}

std::string ROCMOffloadReduceDispatcher::kernel_file_extension() const { return "rocm.cpp"; }

std::string ROCMOffloadReduceDispatcher::warp_shuffle_xor(const std::string& value, const std::string& lane_mask) const {
    // HIP's __shfl_xor_sync requires a 64-bit membership mask (static_assert on
    // sizeof(mask) == 8), so the literal is always ULL-typed. The value covers
    // exactly the physical wavefront: the low 32 bits for wave32 (RDNA), all 64
    // bits for wave64 (CDNA/GCN).
    const std::string mask = rocm_wavefront_size() > 32 ? "0xffffffffffffffffULL" : "0x00000000ffffffffULL";
    return "__shfl_xor_sync(" + mask + ", " + value + ", " + lane_mask + ")";
}


} // namespace rocm
} // namespace sdfg
