#pragma once

#include <string>

#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>

#include "sdfg/analysis/arguments_analysis.h"

namespace sdfg::tenstorrent {

struct TransferArg {
    std::string name;
    const types::IType& type;
    symbolic::Expression data_size;
    symbolic::Expression page_size;
    symbolic::Expression allocated_size;
    analysis::RegionArgument meta;

    TransferArg(
        std::string name,
        const types::IType& type,
        symbolic::Expression data_size,
        symbolic::Expression page_size,
        analysis::RegionArgument meta
    )
        : name(std::move(name)), type(type), data_size(data_size), page_size(page_size),
          allocated_size(calc_allocated_size(data_size, page_size)), meta(meta) {}

    static symbolic::Expression calc_allocated_size(symbolic::Expression& data_size, symbolic::Expression& page_size) {
        auto allocated = symbolic::mul(symbolic::divide_ceil(data_size, page_size), page_size);

        return std::move(allocated);
    }
};

} // namespace sdfg::tenstorrent
