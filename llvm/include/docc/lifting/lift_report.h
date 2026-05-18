#pragma once

#include <llvm/IR/InstrTypes.h>

#include <nlohmann/json.hpp>

#include "sdfg/element.h"

namespace docc {
namespace lifting {

class LiftingReport {
private:
    nlohmann::json report_;
    bool do_reporting_;

    void add_successful_lift_internal(const sdfg::DebugInfo& dbg_info);

    void add_failed_lift_internal(const sdfg::DebugInfo& dbg_info, const std::string& reason, const std::string& element);

    void dump_report_internal(std::filesystem::path report_path);

public:
    LiftingReport();

    static void add_successful_lift(const sdfg::DebugInfo& dbg_info);

    static void add_failed_lift(const sdfg::DebugInfo& dbg_info, const std::string& reason, const std::string& element);

    static void dump_report(std::filesystem::path report_path);
};

} // namespace lifting
} // namespace docc
