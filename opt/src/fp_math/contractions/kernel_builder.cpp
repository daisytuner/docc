#include "sdfg/fp_math/contractions/kernel_builder.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace fp_math {

namespace {

std::string literal(double value, types::PrimitiveType type) {
    if (types::is_integer(type)) {
        return std::to_string(static_cast<long long>(value));
    }
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    auto text = stream.str();
    if (text.find_first_of(".en") == std::string::npos) {
        text += ".0";
    }
    return text;
}

} // namespace

BlockBuilder::
    BlockBuilder(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& block, DebugInfo debug_info)
    : builder_(builder), block_(block), debug_info_(std::move(debug_info)) {
}

KernelValue BlockBuilder::result(types::PrimitiveType type) {
    auto name = builder_.find_new_name("_fpm");
    types::Scalar scalar(type);
    builder_.add_container(name, scalar);
    return {&builder_.add_access(block_, name, debug_info_), {}, std::make_shared<types::Scalar>(scalar)};
}

void BlockBuilder::connect(const KernelValue& value, data_flow::CodeNode& node, const std::string& connector) {
    builder_.add_computational_memlet(block_, *value.node, node, connector, value.subset, *value.base_type, debug_info_);
}

KernelValue BlockBuilder::read(data_flow::AccessNode& node, const data_flow::Subset& subset, const types::IType& base_type) {
    return {&node, subset, std::shared_ptr<types::IType>(base_type.clone())};
}

KernelValue BlockBuilder::read(const std::string& container, const data_flow::Subset& subset) {
    auto& node = builder_.add_access(block_, container, debug_info_);
    return read(node, subset, builder_.subject().type(container));
}

KernelValue BlockBuilder::constant(double value, types::PrimitiveType type) {
    types::Scalar scalar(type);
    auto& node = builder_.add_constant(block_, literal(value, type), scalar, debug_info_);
    return {&node, {}, std::make_shared<types::Scalar>(scalar)};
}

KernelValue BlockBuilder::convert(const KernelValue& value, types::PrimitiveType type) {
    if (value.subset.empty() && value.base_type->primitive_type() == type) {
        return value;
    }
    return op(data_flow::TaskletCode::assign, {value}, type);
}

KernelValue BlockBuilder::
    op(data_flow::TaskletCode code, const std::vector<KernelValue>& inputs, types::PrimitiveType type) {
    std::vector<std::string> connectors;
    for (size_t i = 0; i < inputs.size(); i++) {
        connectors.push_back(inputs.size() == 1 ? "_in" : "_in" + std::to_string(i + 1));
    }
    auto& tasklet = builder_.add_tasklet(block_, code, "_out", connectors, debug_info_);
    for (size_t i = 0; i < inputs.size(); i++) {
        connect(inputs.at(i), tasklet, connectors.at(i));
    }
    auto out = result(type);
    builder_.add_computational_memlet(block_, tasklet, "_out", *out.node, {}, *out.base_type, debug_info_);
    return out;
}

KernelValue BlockBuilder::
    cmath(math::cmath::CMathFunction function, const std::vector<KernelValue>& inputs, types::PrimitiveType type) {
    auto& node = static_cast<math::cmath::CMathNode&>(builder_.add_library_node<
                                                      math::cmath::CMathNode>(block_, debug_info_, function, type));
    for (size_t i = 0; i < inputs.size(); i++) {
        connect(inputs.at(i), node, node.input(i));
    }
    auto out = result(type);
    builder_.add_computational_memlet(block_, node, "_out", *out.node, {}, *out.base_type, debug_info_);
    return out;
}

void BlockBuilder::store(
    const KernelValue& value, data_flow::AccessNode& node, const data_flow::Subset& subset, const types::IType& base_type
) {
    auto& assign = builder_.add_tasklet(block_, data_flow::TaskletCode::assign, "_out", {"_in"}, debug_info_);
    connect(value, assign, "_in");
    builder_.add_computational_memlet(block_, assign, "_out", node, subset, base_type, debug_info_);
}

void BlockBuilder::store(const KernelValue& value, const std::string& container, const data_flow::Subset& subset) {
    auto& node = builder_.add_access(block_, container, debug_info_);
    store(value, node, subset, builder_.subject().type(container));
}

KernelBuilder::KernelBuilder(
    builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& scope, DebugInfo debug_info
)
    : builder_(builder), scope_(scope), debug_info_(std::move(debug_info)) {
}

std::string KernelBuilder::scalar(const std::string& prefix, types::PrimitiveType type) {
    auto name = builder_.find_new_name(prefix);
    builder_.add_container(name, types::Scalar(type));
    return name;
}

std::string KernelBuilder::buffer(const std::string& prefix, types::PrimitiveType type, const symbolic::Expression& count) {
    auto name = builder_.find_new_name(prefix);
    types::Scalar element(type);
    types::Pointer pointer(element);
    builder_.add_container(name, pointer);

    auto bytes = symbolic::mul(count, types::get_type_size(element, false));
    auto& block = builder_.add_block(scope_, {}, debug_info_);
    auto& access = builder_.add_access(block, name, debug_info_);
    auto& malloc = builder_.add_library_node<stdlib::MallocNode>(block, debug_info_, bytes);
    builder_.add_computational_memlet(block, malloc, "_ret", access, {}, pointer, debug_info_);

    buffers_.push_back(name);
    return name;
}

structured_control_flow::StructuredLoop& KernelBuilder::
    map(structured_control_flow::Sequence& scope, const std::string& prefix, const symbolic::Expression& bound) {
    auto indvar = symbolic::symbol(scalar(prefix, types::get_primitive_type_to_hold_upper_bound(bound)));
    return builder_.add_map(
        scope,
        indvar,
        symbolic::Lt(indvar, bound),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        debug_info_
    );
}

structured_control_flow::StructuredLoop& KernelBuilder::
    for_loop(structured_control_flow::Sequence& scope, const std::string& prefix, const symbolic::Expression& bound) {
    auto indvar = symbolic::symbol(scalar(prefix, types::get_primitive_type_to_hold_upper_bound(bound)));
    return builder_.add_for(
        scope, indvar, symbolic::Lt(indvar, bound), symbolic::zero(), symbolic::add(indvar, symbolic::one()), debug_info_
    );
}

structured_control_flow::StructuredLoop& KernelBuilder::reduce(
    structured_control_flow::Sequence& scope,
    const std::string& prefix,
    const symbolic::Expression& bound,
    structured_control_flow::ReductionOperation operation,
    const std::string& accumulator
) {
    auto indvar = symbolic::symbol(scalar(prefix, types::get_primitive_type_to_hold_upper_bound(bound)));
    return builder_.add_reduce(
        scope,
        indvar,
        symbolic::Lt(indvar, bound),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        {{operation, accumulator}},
        structured_control_flow::ScheduleType_Sequential::create(),
        debug_info_
    );
}

BlockBuilder KernelBuilder::block(structured_control_flow::Sequence& scope) {
    return BlockBuilder(builder_, builder_.add_block(scope, {}, debug_info_), debug_info_);
}

void KernelBuilder::finish() {
    for (auto& name : buffers_) {
        auto& block = builder_.add_block(scope_, {}, debug_info_);
        auto& access = builder_.add_access(block, name, debug_info_);
        auto& free = builder_.add_library_node<stdlib::FreeNode>(block, debug_info_);
        builder_.add_computational_memlet(
            block, access, free, "_ptr", {}, types::Pointer(types::Scalar(types::PrimitiveType::Void)), debug_info_
        );
    }
    buffers_.clear();
}

} // namespace fp_math
} // namespace sdfg
