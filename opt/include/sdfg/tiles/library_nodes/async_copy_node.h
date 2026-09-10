#pragma once

#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace tiles {

inline data_flow::LibraryNodeCode LibraryNodeType_CpAsyncCopy{"cp_async_copy"};
inline data_flow::LibraryNodeCode LibraryNodeType_PipelineCommit{"pipeline_commit"};
inline data_flow::LibraryNodeCode LibraryNodeType_PipelineWait{"pipeline_wait"};
inline data_flow::LibraryNodeCode LibraryNodeType_VectorCopy{"vector_copy"};

class CpAsyncCopyNode : public data_flow::LibraryNode {
    size_t bytes_;

public:
    CpAsyncCopyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        size_t bytes
    );

    size_t bytes() const { return bytes_; }
    void set_bytes(size_t bytes) { bytes_ = bytes; }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    /// {"_dst", "_src"}: writes _dst, reads _src, captures neither. Lets escape
    /// analysis treat a container staged through this copy as an ordinary copy so a
    /// later transformation (e.g. a register-accumulator LocalStorage) still applies.
    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class VectorCopyNode : public data_flow::LibraryNode {
    size_t bytes_;

public:
    VectorCopyNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        size_t bytes
    );

    size_t bytes() const { return bytes_; }
    void set_bytes(size_t bytes) { bytes_ = bytes; }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    /// {"_dst", "_src"}: writes _dst, reads _src, captures neither. Lets escape
    /// analysis treat a container staged through this copy as an ordinary copy so a
    /// later transformation (e.g. a register-accumulator LocalStorage) still applies.
    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class PipelineCommitNode : public data_flow::LibraryNode {
public:
    PipelineCommitNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type
    );

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class PipelineWaitNode : public data_flow::LibraryNode {
    size_t keep_outstanding_;
    size_t loads_per_group_;

public:
    PipelineWaitNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::ImplementationType& implementation_type,
        size_t keep_outstanding,
        size_t loads_per_group = 1
    );

    size_t keep_outstanding() const { return keep_outstanding_; }

    size_t loads_per_group() const { return loads_per_group_; }

    void set_loads_per_group(size_t loads_per_group) { loads_per_group_ = loads_per_group; }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

// ---- Serializers ---------------------------------------------------------

class CpAsyncCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class VectorCopyNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineCommitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class PipelineWaitNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

} // namespace tiles
} // namespace sdfg
