#include "sdfg/targets/gpu/gpu_mma_fragment.h"

#include <nlohmann/json.hpp>
#include <sstream>

namespace sdfg::gpu {

void GpuMmaFromMemoryLayout::collect_symbols(symbolic::SymbolSet& symbols) const {
    for (auto& sym : symbolic::atoms(offset)) {
        symbols.insert(sym);
    }
    for (auto& sym : symbolic::atoms(ldstride)) {
        symbols.insert(sym);
    }
}

void GpuMmaFromMemoryLayout::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    offset = symbolic::subs(offset, old_expression, new_expression);
    ldstride = symbolic::subs(ldstride, old_expression, new_expression);
}

void GpuMmaFromMemoryLayout::replace(const symbolic::ExpressionMapping& replacements) {
    offset = symbolic::subs(offset, replacements);
    ldstride = symbolic::subs(ldstride, replacements);
}

std::optional<math::tensor::TensorLayout> GpuMmaFromMemoryLayout::
    to_tensor_layout(const MmaBlockSize& block_size, MmaFragmentType frag) const {
    std::vector<symbolic::Expression> shape = block_size.get_shape(frag);
    std::vector<symbolic::Expression> strides;
    switch (layout) {
        case MmaFragmentLayout::MMA_LAYOUT_ROW_MAJOR:
            strides = {ldstride, symbolic::integer(1)};
            break;
        case MmaFragmentLayout::MMA_LAYOUT_COL_MAJOR:
            strides = {symbolic::integer(1), ldstride};
            break;
        default:
            return std::nullopt;
    }

    return math::tensor::TensorLayout(shape, strides, offset);
}

std::optional<GpuMmaFromMemoryLayout> GpuMmaFromMemoryLayout::from_tensor_layout(const math::tensor::TensorLayout& layout) {
    auto type = layout.is_2d_col_or_row_major();
    if (type == math::tensor::TensorLayout::LAYOUT_ROW_MAJOR) {
        return {{.offset = layout.offset(), .ldstride = layout.get_stride(0), .layout = MMA_LAYOUT_ROW_MAJOR}};
    } else if (type == math::tensor::TensorLayout::LAYOUT_COL_MAJOR) {
        return {{.offset = layout.offset(), .ldstride = layout.get_stride(1), .layout = MMA_LAYOUT_COL_MAJOR}};
    }

    return std::nullopt;
}

std::string MmaBlockSize::dim_str(MmaFragmentType for_type) const {
    switch (for_type) {
        case MmaFragmentType::A:
            return std::to_string(m) + "x" + std::to_string(k);
        case MmaFragmentType::B:
            return std::to_string(k) + "x" + std::to_string(n);
        case MmaFragmentType::C:
            return std::to_string(m) + "x" + std::to_string(n);
        default:
            throw std::invalid_argument("Invalid MMA fragment type");
    }
}

symbolic::MultiExpression MmaBlockSize::get_shape(MmaFragmentType frag) const {
    switch (frag) {
        case MmaFragmentType::A:
            return {symbolic::integer(m), symbolic::integer(k)};
        case MmaFragmentType::B:
            return {symbolic::integer(k), symbolic::integer(n)};
        case MmaFragmentType::C:
            return {symbolic::integer(m), symbolic::integer(n)};
        default:
            throw std::invalid_argument("Invalid MMA fragment type");
    }
}

MmaBlockSize MmaBlockSize::parse_block_size(const std::string& block_size_str) {
    MmaBlockSize block_size{};
    char sep1 = 0, sep2 = 0, extra = 0;
    std::istringstream ss(block_size_str);
    ss >> block_size.m >> sep1 >> block_size.n >> sep2 >> block_size.k;
    if (ss.fail() || (sep1 != 'x' && sep1 != 'X') || (sep2 != 'x' && sep2 != 'X') || (ss >> extra)) {
        throw std::invalid_argument("Invalid MMA block size: '" + block_size_str + "' (expected format MxNxK)");
    }
    return block_size;
}

int GpuMmaSupport::get_storage_type_arg_as_int(const types::StorageType& storage, int idx) {
    auto& entry = storage.args().at(idx);
    auto int_entry = SymEngine::rcp_dynamic_cast<const SymEngine::Integer>(entry);
    if (!int_entry.is_null()) {
        return static_cast<int>(int_entry->as_int());
    } else {
        throw std::runtime_error("Storage type argument at index " + std::to_string(idx) + " is not an integer.");
    }
}

std::string GpuMmaFromMemoryLayout::toStr() const {
    std::stringstream ss;
    ss << "offset: " << offset->__str__() << ", ldstride: " << ldstride->__str__();
    ss << ", " << mma_fragment_layout_to_string(layout);
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const GpuMmaFromMemoryLayout& layout) {
    os << layout.toStr();
    return os;
}

void serialize_mma_block_size(nlohmann::json& j, const MmaBlockSize& block_size) {
    j["m"] = block_size.m;
    j["n"] = block_size.n;
    j["k"] = block_size.k;
}

MmaBlockSize deserialize_mma_block_size(const nlohmann::json& j) {
    return MmaBlockSize{j.at("m").get<int>(), j.at("n").get<int>(), j.at("k").get<int>()};
}

void serialize_mma_from_memory_layout(nlohmann::json& j, const GpuMmaFromMemoryLayout& layout) {
    j["offset"] = layout.offset->__str__();
    j["ldstride"] = layout.ldstride->__str__();
    j["layout"] = static_cast<int>(layout.layout);
}

GpuMmaFromMemoryLayout deserialize_mma_from_memory_layout(const nlohmann::json& j) {
    return GpuMmaFromMemoryLayout{
        symbolic::parse(j.at("offset").get<std::string>()),
        symbolic::parse(j.at("ldstride").get<std::string>()),
        static_cast<MmaFragmentLayout>(j.at("layout").get<int>())
    };
}

} // namespace sdfg::gpu
