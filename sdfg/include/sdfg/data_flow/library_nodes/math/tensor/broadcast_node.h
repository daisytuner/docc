#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Broadcast("ml::Broadcast");

class BroadcastNode : public TensorNode {
private:
    std::vector<symbolic::Expression> input_shape_;
    std::vector<symbolic::Expression> output_shape_;

public:
    BroadcastNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& input_shape,
        const std::vector<symbolic::Expression>& output_shape
    );

    void validate(const Function& function) const override;

    const std::vector<symbolic::Expression>& input_shape() const { return input_shape_; }
    const std::vector<symbolic::Expression>& output_shape() const { return output_shape_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    bool supports_integer_types() const override { return true; }

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class BroadcastNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override {
        const BroadcastNode& reduce_node = static_cast<const BroadcastNode&>(library_node);
        nlohmann::json j;

        j["code"] = reduce_node.code().value();

        serializer::JSONSerializer serializer;
        j["input_shape"] = nlohmann::json::array();
        for (auto& dim : reduce_node.input_shape()) {
            j["input_shape"].push_back(serializer.expression(dim));
        }

        j["output_shape"] = nlohmann::json::array();
        for (auto& dim : reduce_node.output_shape()) {
            j["output_shape"].push_back(serializer.expression(dim));
        }

        return j;
    }

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override {
        // Assertions for required fields
        assert(j.contains("element_id"));
        assert(j.contains("code"));
        assert(j.contains("debug_info"));
        assert(j.contains("input_shape"));
        assert(j.contains("output_shape"));

        auto code = j["code"].get<std::string>();

        std::vector<symbolic::Expression> input_shape;
        for (const auto& dim : j["input_shape"]) {
            input_shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        std::vector<symbolic::Expression> output_shape;
        for (const auto& dim : j["output_shape"]) {
            output_shape.push_back(symbolic::parse(dim.get<std::string>()));
        }

        // Extract debug info using JSONSerializer
        sdfg::serializer::JSONSerializer serializer;
        DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

        return builder.add_library_node<BroadcastNode>(parent, debug_info, input_shape, output_shape);
    }
};

} // namespace tensor
} // namespace math
} // namespace sdfg
