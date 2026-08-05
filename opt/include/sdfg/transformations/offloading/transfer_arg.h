#pragma once

#include <string>
#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/symbolic/symbolic.h"


namespace docc::offloading {

struct TransferArg {
    std::string name;
    const sdfg::types::IType& type;
    sdfg::symbolic::Expression data_size;
    sdfg::analysis::RegionArgument meta;

    TransferArg(
        const std::string& name,
        const sdfg::types::IType& type,
        const sdfg::symbolic::Expression& data_size,
        const sdfg::analysis::RegionArgument& meta
    )
        : name(name), type(type), data_size(data_size), meta(meta) {}
};

} // namespace docc::offloading
