#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_reduce_dispatcher.h"

namespace sdfg {
namespace cuda {

/**
 * @brief Atomics-based CUDA code generator for @ref structured_control_flow::Reduce
 *
 * Thin CUDA specialization of @ref gpu::GPUReduceDispatcher: the whole lowering
 * (grid-stride reduce, privatization, atomic merge) lives in the shared base;
 * this class only supplies the CUDA-specific policy (language extension,
 * device-pointer storage, kernel launch syntax and includes).
 */
class CUDAReduceDispatcher : public gpu::GPUReduceDispatcher {
protected:
    std::string schedule_value() const override;
    codegen::TargetType target_type() const override;
    std::unique_ptr<codegen::LanguageExtension> create_device_language_extension() const override;
    bool is_device_pointer_storage(const types::StorageType& storage) const override;
    std::string kernel_file_extension() const override;
    void emit_kernel_includes(codegen::CodeSnippetFactory& library_snippet_factory) const override;
    void emit_library_preamble(codegen::PrettyPrinter& library_stream) const override;
    void emit_kernel_call(
        codegen::PrettyPrinter& main_stream,
        const std::string& kernel_name,
        symbolic::Expression& num_blocks,
        symbolic::Expression& block_size,
        std::vector<std::string>& arguments_device
    ) override;

public:
    CUDAReduceDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Reduce& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );
};

} // namespace cuda
} // namespace sdfg
