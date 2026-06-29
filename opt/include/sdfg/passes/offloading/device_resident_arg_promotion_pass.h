#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Promotes function pointer arguments to device-resident storage.
 *
 * After offloading, the canonical pattern for every pointer argument is
 *   host_arg --(H2D + malloc)--> dev_buf --kernel--> host_arg (+ D2H + free).
 *
 * When *every* pointer argument is used exclusively by boundary offloading nodes
 * (i.e. no host-side tasklet ever touches it), the whole program can run with
 * device-resident arguments: the caller passes device pointers and the boundary
 * copies degrade to device-to-device copies (handled automatically by the
 * offloading dispatchers, which derive the memcpy kind from the storage tags).
 *
 * This pass flips the storage of all pointer arguments to device storage and
 * records `arg_residency = "device"` / `device_backend` in the SDFG metadata so
 * the runtime knows to pass device pointers. The decision is whole-program and
 * conservative: if any argument is touched by host code, nothing is promoted and
 * the program keeps its original (all-host) behavior.
 */
class DeviceResidentArgPromotionPass : public Pass {
private:
    bool is_rocm_;

public:
    explicit DeviceResidentArgPromotionPass(bool is_rocm);
    ~DeviceResidentArgPromotionPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "DeviceResidentArgPromotionPass"; }
};

} // namespace passes
} // namespace sdfg
