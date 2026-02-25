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
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_TensorCMath("ml::CMath");

class CMathTensorNode : public TensorNode {
private:
    cmath::CMathFunction cmath_function_;
    std::vector<symbolic::Expression> shape_; ///< Logical tensor shape

public:
    CMathTensorNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const cmath::CMathFunction cmath_function,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const std::vector<symbolic::Expression>& shape
    );

    void validate(const Function& function) const override;

    /**
     * @brief Get the operation code
     * @return CMathFunction for this cmath tensor node
     */
    cmath::CMathFunction cmath_function() const;

    /**
     * @brief Get the tensor shape
     * @return Logical tensor shape
     */
    const std::vector<symbolic::Expression>& shape() const;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    /**
     * @brief Expand into map with linearized indexing
     *
     * Creates nested maps over each dimension with linearized index computation
     * for accessing the flat input/output arrays.
     *
     * @param builder SDFG builder
     * @param analysis_manager Analysis manager
     * @return True if expansion succeeded
     */
    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    bool supports_integer_types() const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class CMathTensorNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
