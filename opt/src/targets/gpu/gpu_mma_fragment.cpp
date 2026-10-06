#include "sdfg/targets/gpu/gpu_mma_fragment.h"

#include <nlohmann/json.hpp>

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
