#pragma once

#include "daisy_rtl/primitive_types.h"
#include "sdfg/data_flow/library_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {

namespace data_flow {
inline LibraryNodeCode LibraryNodeType_AtomicScalarOp{"AtomicScalarOp"};

// The implementation_type selects the lowering; the semantics are identical:
// atomically add `_src` into the location `_dst` points at. The concrete
// implementations (AtomicScalarOpImpl) each own their implementation type name.

enum class AtomicOpType { Add, Subtract, Min, Max };

constexpr const char* atomic_op_type_to_string(AtomicOpType type) {
    switch (type) {
        case AtomicOpType::Add:
            return "Add";
        case AtomicOpType::Subtract:
            return "Subtract";
        case AtomicOpType::Min:
            return "Min";
        case AtomicOpType::Max:
            return "Max";
        default:
            throw std::invalid_argument("Invalid AtomicOpType type");
    }
}

constexpr AtomicOpType atomic_op_type_from_string(std::string_view str) {
    if (str == "Add") {
        return AtomicOpType::Add;
    } else if (str == "Subtract") {
        return AtomicOpType::Subtract;
    } else if (str == "Min") {
        return AtomicOpType::Min;
    } else if (str == "Max") {
        return AtomicOpType::Max;
    } else {
        throw std::invalid_argument("Invalid AtomicOpType type: " + std::string(str));
    }
}


class AtomicScalarOpNode;
class AtomicScalarOpNodeDispatcher;

/**
 * @brief A concrete lowering ("implementation") of an AtomicScalarOpNode.
 *
 * An AtomicScalarOpImpl bundles two pieces of knowledge that used to be spread
 * across AtomicScalarOpNode (verify_impl_exits_*) and the dispatchers:
 *  - which (data type, atomic op) combinations it actually supports, and
 *  - how to generate code for a node using this implementation.
 *
 * Each concrete implementation is a stateless singleton, obtained via its
 * instance() method, and is tied to exactly one ImplementationType. This keeps
 * a node's implementation_type in sync with the impl used to reason about and
 * lower it.
 */
class AtomicScalarOpImpl {
public:
    virtual ~AtomicScalarOpImpl() = default;

    AtomicScalarOpImpl(const AtomicScalarOpImpl&) = delete;
    AtomicScalarOpImpl& operator=(const AtomicScalarOpImpl&) = delete;

    virtual std::string_view type_name() const = 0;

    /// The implementation type this impl corresponds to.
    ImplementationType implementation_type() const { return StringEnum(std::string(this->type_name())); }

    /**
     * @brief Whether this implementation supports the given (data type, op) combination.
     */
    virtual bool supports(types::PrimitiveType data_type, AtomicOpType op) const = 0;

protected:
    AtomicScalarOpImpl() {}
};

/**
    Wrapper around typical atomic operations (add, sub, min, max, swap) on a single scalar value. The node is
   parameterized by the data type and the operation type. The implementation type determines how the node is lowered to
   code.
 */
class AtomicScalarOpNode : public LibraryNode {
    types::PrimitiveType data_type_;
    AtomicOpType atomic_op_;
    const AtomicScalarOpImpl* impl_;

public:
    AtomicScalarOpNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        DataFlowGraph& parent,
        types::PrimitiveType data_type,
        AtomicOpType op,
        const AtomicScalarOpImpl* impl
    );

    types::PrimitiveType data_type() const { return data_type_; }
    AtomicOpType atomic_op() const { return atomic_op_; }

    void switch_implementation(const AtomicScalarOpImpl& new_impl);

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, DataFlowGraph& parent)
        const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    void verify_impl_matches() const;

    void verify_impl_exits() const;

    static const AtomicScalarOpImpl* get_implementation(const std::string& type);
};

/// CPU (OpenMP) atomic implementation.
class AtomicScalarOpCPUImpl : public AtomicScalarOpImpl {
public:
    static constexpr const char* TYPE_NAME = "CPU";
    static const AtomicScalarOpCPUImpl* instance();

    std::string_view type_name() const override { return TYPE_NAME; }

    bool supports(types::PrimitiveType data_type, AtomicOpType op) const override;

    static ImplementationType implementation_type() { return {TYPE_NAME}; }

private:
    AtomicScalarOpCPUImpl() : AtomicScalarOpImpl() {}
};

/**
 * Shared base for the GPU implementations (CUDA and ROCm). Their code
 * generation is currently identical; only the supported (data type, op)
 * combinations differ, which the concrete subclasses provide.
 */
class AtomicScalarOpGPUImpl : public AtomicScalarOpImpl {
public:

protected:
    using AtomicScalarOpImpl::AtomicScalarOpImpl;
};

/// CUDA atomic implementation.
class AtomicScalarOpCudaImpl : public AtomicScalarOpGPUImpl {
public:
    static constexpr const char* TYPE_NAME = "CUDA";
    static const AtomicScalarOpCudaImpl* instance();

    std::string_view type_name() const override { return TYPE_NAME; }

    bool supports(types::PrimitiveType data_type, AtomicOpType op) const override;

    static ImplementationType implementation_type() { return {TYPE_NAME}; }

private:
    AtomicScalarOpCudaImpl() : AtomicScalarOpGPUImpl() {}
};

/// ROCm atomic implementation.
class AtomicScalarOpRocmImpl : public AtomicScalarOpGPUImpl {
public:
    static constexpr const char* TYPE_NAME = "ROCm";
    static const AtomicScalarOpRocmImpl* instance();

    std::string_view type_name() const override { return TYPE_NAME; }

    bool supports(types::PrimitiveType data_type, AtomicOpType op) const override;

    static ImplementationType implementation_type() { return {TYPE_NAME}; }

private:
    AtomicScalarOpRocmImpl() : AtomicScalarOpGPUImpl() {}
};

class AtomicScalarOpNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) override;
};

class AtomicScalarOpNodeDispatcher : public codegen::LibraryNodeDispatcher {
protected:
    AtomicScalarOpNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::AtomicScalarOpNode& node
    );
};

class AtomicScalarOpCPUNodeDispatcher : public AtomicScalarOpNodeDispatcher {
public:
    AtomicScalarOpCPUNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::AtomicScalarOpNode& node
    )
        : AtomicScalarOpNodeDispatcher(language_extension, function, data_flow_graph, node) {}

    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};

class AtomicScalarOpGPUNodeDispatcher : public AtomicScalarOpNodeDispatcher {
public:
    AtomicScalarOpGPUNodeDispatcher(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::AtomicScalarOpNode& node
    )
        : AtomicScalarOpNodeDispatcher(language_extension, function, data_flow_graph, node) {}

public:
    void dispatch_code_with_edges(
        codegen::CodegenOutput& out,
        std::vector<codegen::DispatchInput>& inputs,
        std::vector<codegen::DispatchOutput>& outputs
    ) override;
};


} // namespace data_flow
} // namespace sdfg
