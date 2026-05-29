#pragma once

#include <cstddef>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_TensorTasklet("ml::Tasklet");

class TaskletTensorNode : public ElementWiseDataflowTensorNode {
private:
    data_flow::TaskletCode tasklet_code_;

public:
    TaskletTensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::TaskletCode tasklet_code,
        const std::string& modified_tensor_conn,
        const std::vector<std::string>& tensor_inputs,
        const std::vector<symbolic::Expression>& shape,
        QuantizationType quantization = QUANTIZATION_MATCH_INPUTS,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    void validate(const Function& function) const override;

    /**
     * @brief Get the operation code
     * @return TaskletCode for this tasklet tensor node
     */
    data_flow::TaskletCode tasklet_code() const;

    bool supports_integer_types() const override;

    ElementOutput expand_operation_dataflow(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        Block& block,
        std::vector<ElementInput>& needed_inputs,
        types::PrimitiveType expected_type
    ) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class TaskletTensorNodeSerializer : public BaseElementWiseDataflowTensorNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
