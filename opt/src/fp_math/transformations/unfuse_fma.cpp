#include "sdfg/fp_math/transformations/unfuse_fma.h"

#include <nlohmann/json.hpp>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace fp_math {

UnfuseFMA::UnfuseFMA(data_flow::Tasklet& fma) : fma_(fma) {
}

std::string UnfuseFMA::name() const {
    return "UnfuseFMA";
}

bool UnfuseFMA::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = fma_.get_parent();
    if (fma_.code() != data_flow::TaskletCode::fp_fma ||
        !dynamic_cast<structured_control_flow::Block*>(dataflow.get_parent())) {
        return false;
    }
    if (dataflow.in_degree(fma_) != 3 || dataflow.out_degree(fma_) != 1) {
        return false;
    }
    auto type = (*dataflow.out_edges(fma_).begin()).result_type(builder.subject());
    return type && type->type_id() == types::TypeID::Scalar && types::is_floating_point(type->primitive_type());
}

void UnfuseFMA::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = fma_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto inputs = dataflow.in_edges_by_connector(fma_);
    auto& output = *dataflow.out_edges(fma_).begin();
    types::Scalar element(output.result_type(builder.subject())->primitive_type());

    product_ = builder.find_new_name("_fma_product");
    builder.add_container(product_, element);

    auto& multiply =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, fma_.debug_info());
    auto& add = builder.add_tasklet(block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, fma_.debug_info());
    auto& product_access = builder.add_access(block, product_, fma_.debug_info());

    auto rewire = [&](data_flow::Memlet& memlet, data_flow::CodeNode& consumer, const std::string& connector) {
        builder.add_memlet(
            block,
            memlet.src(),
            memlet.src_conn(),
            consumer,
            connector,
            memlet.subset(),
            memlet.base_type(),
            memlet.debug_info()
        );
    };
    rewire(*inputs.at(0), multiply, "_in1");
    rewire(*inputs.at(1), multiply, "_in2");
    rewire(*inputs.at(2), add, "_in2");
    builder.add_computational_memlet(block, multiply, "_out", product_access, {}, element, fma_.debug_info());
    builder.add_computational_memlet(block, product_access, add, "_in1", {}, element, fma_.debug_info());
    builder.add_memlet(
        block, add, "_out", output.dst(), output.dst_conn(), output.subset(), output.base_type(), output.debug_info()
    );

    std::vector<data_flow::Memlet*> obsolete{&output};
    obsolete.insert(obsolete.end(), inputs.begin(), inputs.end());
    for (auto* memlet : obsolete) {
        builder.remove_memlet(block, *memlet);
    }
    builder.remove_node(block, fma_);

    multiply_ = &multiply;
    add_ = &add;
    analysis_manager.invalidate_all();
}

void UnfuseFMA::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->fma_.element_id()}, {"type", "tasklet"}}}};
}

UnfuseFMA UnfuseFMA::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* tasklet = dynamic_cast<data_flow::Tasklet*>(builder.find_element_by_id(id));
    if (!tasklet) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(id) + " is not a tasklet");
    }
    return UnfuseFMA(*tasklet);
}

data_flow::Tasklet* UnfuseFMA::multiply() const {
    return multiply_;
}

data_flow::Tasklet* UnfuseFMA::add() const {
    return add_;
}

const std::string& UnfuseFMA::product() const {
    return product_;
}

} // namespace fp_math
} // namespace sdfg
