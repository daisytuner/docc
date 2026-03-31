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
#include "sdfg/data_flow/library_nodes/math/math_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/element.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace einsum {

inline data_flow::LibraryNodeCode LibraryNodeType_Einsum("Einsum");

struct EinsumDimension {
    symbolic::Symbol indvar;
    symbolic::Expression init;
    symbolic::Expression bound;
};

class EinsumNode : public math::MathNode {
private:
    std::vector<EinsumDimension> dims_;

    data_flow::Subset out_indices_;
    std::vector<data_flow::Subset> in_indices_;

public:
    EinsumNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<std::string>& inputs,
        const std::vector<EinsumDimension>& dims,
        const data_flow::Subset& out_indices,
        const std::vector<data_flow::Subset>& in_indices,
        bool rename_indvars = true
    );

    EinsumNode(const EinsumNode&) = delete;
    EinsumNode& operator=(const EinsumNode&) = delete;

    virtual ~EinsumNode() = default;

    const std::vector<EinsumDimension>& dims() const;

    const EinsumDimension& dim(size_t index) const;

    const symbolic::Symbol& indvar(size_t index) const;

    const symbolic::Expression& init(size_t index) const;

    const symbolic::Expression& bound(size_t index) const;

    const data_flow::Subset& out_indices() const;

    const symbolic::Expression& out_index(size_t index) const;

    const std::vector<data_flow::Subset>& in_indices() const;

    const data_flow::Subset& in_indices(size_t index) const;

    const symbolic::Expression& in_index(size_t index1, size_t index2) const;

    symbolic::SymbolSet internal_symbols() const;

    virtual bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual symbolic::SymbolSet symbols() const override;

    virtual void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    virtual std::string toStr() const override;

    virtual symbolic::Expression flop() const override;

    virtual std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    virtual void validate(const Function& function) const override;
};

class EinsumSerializer : public serializer::LibraryNodeSerializer {
public:
    virtual nlohmann::json serialize(const data_flow::LibraryNode& libnode) override;

    virtual data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace einsum
} // namespace sdfg
