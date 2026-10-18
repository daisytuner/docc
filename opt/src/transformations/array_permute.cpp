#include "sdfg/transformations/array_permute.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <optional>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/copy_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/polynomials.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/utils.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace transformations {

namespace {

std::optional<int64_t> constant(const symbolic::Expression& expr) {
    auto expanded = symbolic::expand(expr);
    if (!SymEngine::is_a<SymEngine::Integer>(*expanded)) {
        return std::nullopt;
    }
    try {
        return SymEngine::down_cast<const SymEngine::Integer&>(*expanded).as_int();
    } catch (const SymEngine::SymEngineException&) {
        return std::nullopt;
    }
}

/// The values [min, max] of the induction variables of the loops around @p node with constant ranges.
std::map<std::string, std::pair<int64_t, int64_t>> ranges_around(structured_control_flow::ControlFlowNode& node) {
    std::map<std::string, std::pair<int64_t, int64_t>> ranges;
    for (auto* ancestor = node.get_parent(); ancestor; ancestor = ancestor->get_parent()) {
        auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(ancestor);
        if (!loop || !loop->is_contiguous()) {
            continue;
        }
        auto first = constant(loop->init());
        auto bound = loop->canonical_bound();
        auto end = bound.is_null() ? std::nullopt : constant(bound);
        if (first && end && *first < *end) {
            ranges[loop->indvar()->get_name()] = {*first, *end - 1};
        }
    }
    return ranges;
}

/// The coordinates of flat @p index in a row-major tensor of @p shape, if every term of the affine index, and its
/// constant, falls into one dimension and stays within its extent over @p ranges.
std::optional<std::vector<symbolic::Expression>> coordinates(
    const symbolic::Expression& index,
    const std::vector<int64_t>& shape,
    const std::map<std::string, std::pair<int64_t, int64_t>>& ranges
) {
    auto expanded = symbolic::expand(index);
    if (!symbolic::is_integer_affine(expanded)) {
        return std::nullopt;
    }
    size_t dims = shape.size();
    std::vector<int64_t> weights(dims, 1);
    for (size_t k = dims - 1; k-- > 0;) {
        weights[k] = weights[k + 1] * shape[k + 1];
    }
    symbolic::ExpressionMapping zeros;
    for (auto& atom : symbolic::atoms(expanded)) {
        zeros[atom] = symbolic::zero();
    }
    auto offset = constant(symbolic::subs(expanded, zeros));
    if (!offset) {
        return std::nullopt;
    }

    std::vector<symbolic::Expression> terms(dims, symbolic::zero());
    std::vector<int64_t> lows(dims, 0);
    std::vector<int64_t> highs(dims, 0);
    for (auto& atom : symbolic::atoms(expanded)) {
        auto range = ranges.find(atom->get_name());
        auto with_one = zeros;
        with_one[atom] = symbolic::one();
        auto coefficient = constant(symbolic::sub(symbolic::subs(expanded, with_one), symbolic::integer(*offset)));
        if (range == ranges.end() || !coefficient || *coefficient == 0) {
            return std::nullopt;
        }
        // The outermost dimension whose weight divides the coefficient.
        size_t k = 0;
        while (*coefficient % weights[k] != 0) {
            ++k;
        }
        auto scaled = *coefficient / weights[k];
        terms[k] = symbolic::add(terms[k], symbolic::mul(symbolic::integer(scaled), atom));
        lows[k] += scaled * (scaled > 0 ? range->second.first : range->second.second);
        highs[k] += scaled * (scaled > 0 ? range->second.second : range->second.first);
    }

    // The constant goes, innermost dimension first, where it keeps each coordinate within its extent.
    auto remaining = *offset;
    for (size_t k = dims; k-- > 0;) {
        auto units = remaining / weights[k];
        int64_t part = units;
        if (k > 0) {
            auto shifted = (units + lows[k]) % shape[k];
            part = (shifted < 0 ? shifted + shape[k] : shifted) - lows[k];
        }
        if (lows[k] + part < 0 || highs[k] + part >= shape[k]) {
            return std::nullopt;
        }
        remaining -= part * weights[k];
        terms[k] = symbolic::add(terms[k], symbolic::integer(part));
    }
    return terms;
}

bool is_bookkeeping(data_flow::Memlet& memlet) {
    auto other = dynamic_cast<data_flow::LibraryNode*>(&memlet.src());
    if (!other) {
        other = dynamic_cast<data_flow::LibraryNode*>(&memlet.dst());
    }
    return memlet.subset().empty() && other &&
           (dynamic_cast<stdlib::MallocNode*>(other) || dynamic_cast<stdlib::FreeNode*>(other));
}

struct Access {
    structured_control_flow::Block* block;
    data_flow::Memlet* memlet;
    data_flow::AccessNode* node;
    bool reads;
    bool in_region;
};

using Box = std::vector<std::pair<int64_t, int64_t>>;

/// The box [low, high] per dimension that the coordinates @p coords take over @p ranges, if they take each of its
/// points: each coordinate a mixed radix of its loops, the first of stride 1, each next of the previous' range.
std::optional<Box> box_of(
    const std::vector<symbolic::Expression>& coords, const std::map<std::string, std::pair<int64_t, int64_t>>& ranges
) {
    Box box;
    for (auto& coordinate : coords) {
        symbolic::ExpressionMapping zeros;
        for (auto& atom : symbolic::atoms(coordinate)) {
            zeros[atom] = symbolic::zero();
        }
        auto low = constant(symbolic::subs(coordinate, zeros));
        if (!low) {
            return std::nullopt;
        }
        std::vector<std::pair<int64_t, int64_t>> digits; // coefficient, extent
        for (auto& atom : symbolic::atoms(coordinate)) {
            auto with_one = zeros;
            with_one[atom] = symbolic::one();
            auto coefficient = constant(symbolic::sub(symbolic::subs(coordinate, with_one), symbolic::integer(*low)));
            auto range = ranges.find(atom->get_name());
            if (!coefficient || *coefficient <= 0 || range == ranges.end()) {
                return std::nullopt;
            }
            *low += *coefficient * range->second.first;
            digits.emplace_back(*coefficient, range->second.second - range->second.first + 1);
        }
        std::ranges::sort(digits);
        int64_t stride = 1;
        for (auto& [coefficient, extent] : digits) {
            if (coefficient != stride) {
                return std::nullopt;
            }
            stride *= extent;
        }
        box.emplace_back(*low, *low + stride - 1);
    }
    return box;
}

/// Whether the unconditional writes among @p accesses take every element of a tensor of @p shape, each once.
bool writes_all(
    const std::vector<Access>& accesses, const std::vector<int64_t>& shape, structured_control_flow::Sequence& parent
) {
    std::vector<Box> boxes;
    for (auto& access : accesses) {
        if (access.reads) {
            continue;
        }
        for (auto* ancestor = access.block->get_parent(); ancestor && ancestor != &parent;
             ancestor = ancestor->get_parent()) {
            if (dynamic_cast<structured_control_flow::IfElse*>(ancestor) ||
                dynamic_cast<structured_control_flow::While*>(ancestor)) {
                return false;
            }
        }
        auto ranges = ranges_around(*access.block);
        auto coords = coordinates(access.memlet->subset().front(), shape, ranges);
        auto box = coords ? box_of(*coords, ranges) : std::nullopt;
        if (!box) {
            return false;
        }
        boxes.push_back(*box);
    }
    int64_t volume = 0;
    for (size_t a = 0; a < boxes.size(); ++a) {
        int64_t size = 1;
        for (auto& [low, high] : boxes[a]) {
            size *= high - low + 1;
        }
        volume += size;
        for (size_t b = 0; b < a; ++b) {
            bool overlap = true;
            for (size_t k = 0; k < shape.size(); ++k) {
                overlap &= boxes[a][k].first <= boxes[b][k].second && boxes[b][k].first <= boxes[a][k].second;
            }
            if (overlap) {
                return false;
            }
        }
    }
    return volume == std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<>());
}

} // namespace

ArrayPermute::ArrayPermute(
    structured_control_flow::Sequence& parent,
    structured_control_flow::ControlFlowNode& first,
    structured_control_flow::ControlFlowNode& last,
    const std::string& container,
    const std::vector<int64_t>& shape,
    const std::vector<size_t>& permutation
)
    : parent_(parent), first_(first), last_(last), container_(container), shape_(shape), permutation_(permutation),
      permuted_(container) {
}

std::string ArrayPermute::name() const {
    return "ArrayPermute";
}

namespace {

/// The accesses to @p container in and outside the children of @p parent from @p first to @p last.
std::pair<std::vector<Access>, std::vector<Access>> accesses(
    StructuredSDFG& sdfg,
    structured_control_flow::Sequence& parent,
    structured_control_flow::ControlFlowNode& first,
    structured_control_flow::ControlFlowNode& last,
    const std::string& container
) {
    auto begin = parent.index(first);
    auto end = parent.index(last);
    auto in_region = [&](structured_control_flow::ControlFlowNode& node) {
        structured_control_flow::ControlFlowNode* child = &node;
        while (child->get_parent() && child->get_parent() != &parent) {
            child = child->get_parent();
        }
        auto index = child->get_parent() ? parent.index(*child) : -1;
        return index >= begin && index <= end;
    };
    std::pair<std::vector<Access>, std::vector<Access>> result;
    visitor::for_each_block(sdfg.root(), [&](structured_control_flow::Block& block) {
        bool inside = in_region(block);
        for (auto& memlet : block.dataflow().edges()) {
            for (bool reads : {true, false}) {
                auto node = dynamic_cast<data_flow::AccessNode*>(reads ? &memlet.src() : &memlet.dst());
                if (node && node->data() == container) {
                    // Allocating or freeing the region's own container is no access to its elements.
                    (inside && !is_bookkeeping(memlet) ? result.first : result.second)
                        .push_back({&block, &memlet, node, reads, inside});
                }
            }
        }
    });
    return result;
}

const types::Scalar* element_of(const types::IType& type) {
    if (auto pointer = dynamic_cast<const types::Pointer*>(&type)) {
        return pointer->has_pointee_type() ? dynamic_cast<const types::Scalar*>(&pointer->pointee_type()) : nullptr;
    }
    if (auto array = dynamic_cast<const types::Array*>(&type)) {
        return dynamic_cast<const types::Scalar*>(&array->element_type());
    }
    return nullptr;
}

/// Where the program uses a container apart from the region's accesses to its elements.
struct Uses {
    bool before = false; ///< elements accessed before the region, or in a loop around it
    bool after = false; ///< elements accessed after the region, or in a loop around it
    bool allocated_inside = false;
    bool freed_inside = false;
    bool in_loop = false; ///< the region runs repeatedly
};

Uses uses_of(
    const std::vector<Access>& outside,
    structured_control_flow::Sequence& parent,
    structured_control_flow::ControlFlowNode& first
) {
    Uses uses;
    for (auto* ancestor = parent.get_parent(); ancestor; ancestor = ancestor->get_parent()) {
        uses.in_loop |= dynamic_cast<structured_control_flow::StructuredLoop*>(ancestor) ||
                        dynamic_cast<structured_control_flow::While*>(ancestor);
    }
    for (auto& access : outside) {
        if (is_bookkeeping(*access.memlet)) {
            if (access.in_region) {
                (dynamic_cast<stdlib::MallocNode*>(&access.memlet->src()) ? uses.allocated_inside
                                                                          : uses.freed_inside) = true;
            }
            continue;
        }
        structured_control_flow::ControlFlowNode* child = access.block;
        while (child->get_parent() && child->get_parent() != &parent) {
            child = child->get_parent();
        }
        if (!child->get_parent()) {
            uses.before = uses.after = true;
        } else {
            (parent.index(*child) < parent.index(first) ? uses.before : uses.after) = true;
        }
    }
    return uses;
}

} // namespace

bool ArrayPermute::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    if (!sdfg.exists(container_) || !element_of(sdfg.type(container_)) || shape_.empty() ||
        shape_.size() != permutation_.size() || parent_.index(first_) < 0 ||
        parent_.index(last_) < parent_.index(first_)) {
        return false;
    }
    std::vector<size_t> sorted(permutation_);
    std::ranges::sort(sorted);
    std::vector<size_t> identity(shape_.size());
    std::iota(identity.begin(), identity.end(), 0);
    if (sorted != identity || permutation_ == identity || std::ranges::any_of(shape_, [](int64_t extent) {
            return extent < 1;
        })) {
        return false;
    }
    auto [inside, outside] = accesses(sdfg, parent_, first_, last_, container_);
    if (inside.empty()) {
        return false;
    }
    // Copies around the region cannot reach a container the region allocates or frees, except to hand on what it
    // holds when the region ends, from an allocation within it.
    auto uses = uses_of(outside, parent_, first_);
    bool copies = uses.before || uses.after || !sdfg.is_transient(container_);
    if ((uses.allocated_inside && uses.before) || (uses.freed_inside && uses.after) ||
        ((uses.allocated_inside || uses.freed_inside) && copies && (uses.in_loop || !sdfg.is_transient(container_)))) {
        return false;
    }
    for (auto& access : inside) {
        auto other = access.reads ? &access.memlet->dst() : &access.memlet->src();
        if (dynamic_cast<data_flow::AccessNode*>(other) || access.memlet->subset().size() != 1 ||
            !coordinates(access.memlet->subset().front(), shape_, ranges_around(*access.block))) {
            return false;
        }
    }
    return true;
}

void ArrayPermute::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = builder.subject();
    auto [inside, outside] = accesses(sdfg, parent_, first_, last_, container_);
    size_t dims = shape_.size();
    std::vector<int64_t> permuted_shape(dims);
    for (size_t j = 0; j < dims; ++j) {
        permuted_shape[j] = shape_[permutation_[j]];
    }
    // Row-major strides of the tensor, and of each of its dimensions where the permutation stores it.
    std::vector<int64_t> strides(dims, 1);
    std::vector<int64_t> permuted_strides(dims, 1);
    std::vector<int64_t> strides_after(dims, 1);
    for (size_t k = dims - 1; k-- > 0;) {
        strides[k] = strides[k + 1] * shape_[k + 1];
        permuted_strides[k] = permuted_strides[k + 1] * permuted_shape[k + 1];
    }
    for (size_t j = 0; j < dims; ++j) {
        strides_after[permutation_[j]] = permuted_strides[j];
    }

    bool transient = sdfg.is_transient(container_);
    auto uses = uses_of(outside, parent_, first_);
    bool in_place = transient && !uses.before && !uses.after;
    bool reads = std::ranges::any_of(inside, [](auto& access) {
        return access.reads;
    });
    // A copy the region writes only in part must hold the other elements to copy back.
    bool copy_in = (!transient || uses.in_loop || uses.before) && (reads || !writes_all(inside, shape_, parent_));
    bool copy_out = std::ranges::any_of(
                        inside,
                        [](auto& access) {
                            return !access.reads;
                        }
                    ) &&
                    (!transient || uses.in_loop || uses.after);

    for (auto& access : inside) {
        auto coords = *coordinates(access.memlet->subset().front(), shape_, ranges_around(*access.block));
        symbolic::Expression index = symbolic::zero();
        for (size_t k = 0; k < dims; ++k) {
            index = symbolic::add(index, symbolic::mul(coords[k], symbolic::integer(strides_after[k])));
        }
        access.memlet->set_subset({symbolic::expand(index)});
    }
    if (in_place) {
        analysis_manager.invalidate_all();
        return;
    }

    auto& element = *element_of(sdfg.type(container_));
    permuted_ = builder.find_new_name(container_ + "_p");
    types::Pointer pointer(element);
    builder.add_container(permuted_, pointer);
    for (auto& access : inside) {
        access.node->data(permuted_);
    }

    auto count = std::accumulate(shape_.begin(), shape_.end(), int64_t{1}, std::multiplies<>());
    auto to_expressions = [](const std::vector<int64_t>& values) {
        symbolic::MultiExpression result;
        for (auto value : values) {
            result.push_back(symbolic::integer(value));
        }
        return result;
    };
    auto add_copy = [&](structured_control_flow::Block& block,
                        const std::string& source,
                        const math::tensor::TensorLayout& layout_x,
                        const std::string& target,
                        const math::tensor::TensorLayout& layout_y) {
        auto& x = builder.add_access(block, source);
        auto& y = builder.add_access(block, target);
        auto& copy = builder.add_library_node<math::tensor::TensorCopyNode>(block, DebugInfo(), layout_x, layout_y);
        builder.add_computational_memlet(block, x, copy, "X", {}, types::Tensor(element, layout_x));
        builder.add_computational_memlet(block, y, copy, "Y", {}, types::Tensor(element, layout_y));
    };

    auto& allocation = builder.add_block_before(parent_, first_, first_.debug_info());
    auto& allocated = builder.add_access(allocation, permuted_);
    auto& malloc = builder.add_library_node<stdlib::MallocNode>(
        allocation, DebugInfo(), symbolic::mul(symbolic::integer(count), types::get_type_size(element, false))
    );
    builder.add_computational_memlet(allocation, malloc, "_ret", allocated, {}, pointer);
    if (copy_in) {
        auto& copy_in_block = builder.add_block_before(parent_, first_, first_.debug_info());
        std::vector<int64_t> source_strides(dims);
        for (size_t j = 0; j < dims; ++j) {
            source_strides[j] = strides[permutation_[j]];
        }
        add_copy(
            copy_in_block,
            container_,
            math::tensor::TensorLayout(to_expressions(permuted_shape), to_expressions(source_strides)),
            permuted_,
            math::tensor::TensorLayout(to_expressions(permuted_shape))
        );
    }
    auto& release = builder.add_block_after(parent_, last_, last_.debug_info());
    auto& released = builder.add_access(release, permuted_);
    auto& free = builder.add_library_node<stdlib::FreeNode>(release, DebugInfo());
    builder.add_computational_memlet(
        release, released, free, "_ptr", {}, types::Pointer(types::Scalar(types::PrimitiveType::Void))
    );
    if (copy_out) {
        auto& copy_out_block = builder.add_block_after(parent_, last_, last_.debug_info());
        add_copy(
            copy_out_block,
            permuted_,
            math::tensor::TensorLayout(to_expressions(shape_), to_expressions(strides_after)),
            container_,
            math::tensor::TensorLayout(to_expressions(shape_))
        );
    }
    analysis_manager.invalidate_all();
}

void ArrayPermute::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = {{"container", container_}, {"shape", shape_}, {"permutation", permutation_}};
    serializer::JSONSerializer serializer(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["0"], first_);
    j["subgraph"]["1"] = nlohmann::json::object();
    serializer.serialize_node(j["subgraph"]["1"], last_);
}

ArrayPermute ArrayPermute::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto find = [&](const char* key) {
        auto id = desc["subgraph"][key]["element_id"].get<size_t>();
        auto node = dynamic_cast<structured_control_flow::ControlFlowNode*>(builder.find_element_by_id(id));
        if (!node || !dynamic_cast<structured_control_flow::Sequence*>(node->get_parent())) {
            throw InvalidTransformationDescriptionException("Element with ID " + std::to_string(id) + " not found.");
        }
        return node;
    };
    auto first = find("0");
    auto last = find("1");
    auto& parameters = desc["parameters"];
    return ArrayPermute(
        static_cast<structured_control_flow::Sequence&>(*first->get_parent()),
        *first,
        *last,
        parameters["container"].get<std::string>(),
        parameters["shape"].get<std::vector<int64_t>>(),
        parameters["permutation"].get<std::vector<size_t>>()
    );
}

const std::string& ArrayPermute::permuted_container() const {
    return permuted_;
}

} // namespace transformations
} // namespace sdfg
