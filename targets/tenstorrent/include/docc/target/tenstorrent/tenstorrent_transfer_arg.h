#pragma once

#include <string>

#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/transformations/offloading/transfer_arg.h"


namespace sdfg::tenstorrent {

struct TransferArg : public docc::offloading::TransferArg {
    symbolic::Expression page_size;
    symbolic::Expression allocated_size;

    TransferArg(
        const std::string& name,
        const types::IType& type,
        const symbolic::Expression& data_size,
        const symbolic::Expression& page_size,
        const analysis::RegionArgument& meta
    )
        : docc::offloading::TransferArg(name, type, data_size, meta), page_size(page_size),
          allocated_size(calc_allocated_size(data_size, page_size)) {}

    static symbolic::Expression
    calc_allocated_size(const symbolic::Expression& data_size, const symbolic::Expression& page_size) {
        auto allocated = symbolic::mul(symbolic::divide_ceil(data_size, page_size), page_size);

        return std::move(allocated);
    }
};

} // namespace sdfg::tenstorrent
