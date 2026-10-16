#include "sdfg/fp_math/transformations/fuse_fma.h"

#include <nlohmann/json.hpp>

#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace fp_math {

namespace {

/// The primitive type @p memlet carries if it is a floating-point scalar, None otherwise.
types::PrimitiveType scalar_fp_type(const data_flow::Memlet& memlet, const Function& sdfg) {
    auto type = memlet.result_type(sdfg);
    if (!type || type->type_id() != types::TypeID::Scalar || !types::is_floating_point(type->primitive_type())) {
        return types::PrimitiveType::Void;
    }
    return type->primitive_type();
}

} // namespace

FuseFMA::FuseFMA(data_flow::Tasklet& add) : add_(add) {
}

std::string FuseFMA::name() const {
    return "FuseFMA";
}

bool FuseFMA::match(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, Match& out) {
    auto& sdfg = builder.subject();
    auto& dataflow = add_.get_parent();
    if (add_.code() != data_flow::TaskletCode::fp_add ||
        !dynamic_cast<structured_control_flow::Block*>(dataflow.get_parent())) {
        return false;
    }
    if (dataflow.in_degree(add_) != 2 || dataflow.out_degree(add_) != 1) {
        return false;
    }
    auto precision = scalar_fp_type(*dataflow.out_edges(add_).begin(), sdfg);
    if (precision == types::PrimitiveType::Void) {
        return false;
    }

    auto inputs = dataflow.in_edges_by_connector(add_);
    for (size_t i = 0; i < inputs.size(); i++) {
        auto* product_edge = inputs.at(i);
        auto* addend_edge = inputs.at(1 - i);
        if (scalar_fp_type(*product_edge, sdfg) != precision || scalar_fp_type(*addend_edge, sdfg) != precision) {
            return false;
        }

        // The product: a transient scalar computed by a multiplication and consumed only by the addition.
        auto* product = dynamic_cast<data_flow::AccessNode*>(&product_edge->src());
        if (!product || dynamic_cast<data_flow::ConstantNode*>(product)) {
            continue;
        }
        if (dataflow.in_degree(*product) != 1 || dataflow.out_degree(*product) != 1) {
            continue;
        }
        auto& product_type = sdfg.type(product->data());
        if (product_type.type_id() != types::TypeID::Scalar || product_type.primitive_type() != precision ||
            !sdfg.is_transient(product->data())) {
            continue;
        }
        auto& users = analysis_manager.get<analysis::Users>();
        bool local = true;
        for (auto* user : users.uses(product->data())) {
            if (user->element() != product) {
                local = false;
                break;
            }
        }
        if (!local) {
            continue;
        }

        auto& write = *dataflow.in_edges(*product).begin();
        auto* multiply = dynamic_cast<data_flow::Tasklet*>(&write.src());
        if (!multiply || multiply->code() != data_flow::TaskletCode::fp_mul || dataflow.in_degree(*multiply) != 2 ||
            dataflow.out_degree(*multiply) != 1 || scalar_fp_type(write, sdfg) != precision) {
            continue;
        }
        bool same_precision = true;
        for (auto& factor : dataflow.in_edges(*multiply)) {
            same_precision &= scalar_fp_type(factor, sdfg) == precision;
        }
        if (!same_precision) {
            continue;
        }

        out = Match{multiply, product, product_edge, addend_edge};
        return true;
    }
    return false;
}

bool FuseFMA::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    Match match;
    return this->match(builder, analysis_manager, match);
}

void FuseFMA::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    Match match;
    if (!this->match(builder, analysis_manager, match)) {
        throw transformations::InvalidTransformationException("FuseFMA: transformation cannot be applied");
    }
    auto& dataflow = add_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    auto& output = *dataflow.out_edges(add_).begin();
    auto factors = dataflow.in_edges_by_connector(*match.multiply);
    auto& product_write = *dataflow.in_edges(*match.product).begin();

    auto debug_info = DebugInfo::merge(match.multiply->debug_info(), add_.debug_info());
    auto& fma =
        builder.add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"}, debug_info);

    auto rewire = [&](data_flow::Memlet& memlet, const std::string& connector) {
        builder.add_memlet(
            block,
            memlet.src(),
            memlet.src_conn(),
            fma,
            connector,
            memlet.subset(),
            memlet.base_type(),
            memlet.debug_info()
        );
    };
    rewire(*factors.at(0), "_in1");
    rewire(*factors.at(1), "_in2");
    rewire(*match.addend_edge, "_in3");
    builder.add_memlet(
        block, fma, "_out", output.dst(), output.dst_conn(), output.subset(), output.base_type(), output.debug_info()
    );

    std::vector<data_flow::Memlet*> obsolete{
        factors.at(0), factors.at(1), &product_write, match.product_edge, match.addend_edge, &output
    };
    for (auto* memlet : obsolete) {
        builder.remove_memlet(block, *memlet);
    }
    auto product = match.product->data();
    builder.remove_node(block, *match.multiply);
    builder.remove_node(block, *match.product);
    builder.remove_node(block, add_);
    builder.remove_container(product);

    fma_ = &fma;
    analysis_manager.invalidate_all();
}

void FuseFMA::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->add_.element_id()}, {"type", "tasklet"}}}};
}

FuseFMA FuseFMA::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* tasklet = dynamic_cast<data_flow::Tasklet*>(builder.find_element_by_id(id));
    if (!tasklet) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(id) + " is not a tasklet");
    }
    return FuseFMA(*tasklet);
}

data_flow::Tasklet* FuseFMA::fma() const {
    return fma_;
}

} // namespace fp_math
} // namespace sdfg
