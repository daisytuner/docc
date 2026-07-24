#pragma once
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"

namespace sdfg::passes {

class LibNodeExpander {
public:
    enum class InputUse { Skip = 0, Scalar, IndirectRead, IndirectWrite, IndirectReadWrite };

    struct ExpandOutcome {
        bool expanded;
        explicit ExpandOutcome(bool expanded) : expanded(expanded) {}
    };
    virtual ~LibNodeExpander() = default;

    /**
     * Prefiltering of libnodes.
     *
     * Allows that an expander sub-divides based on types, codes or impl-types or sth. more elaborate, because we get
     * ready to transform
     *
     * @param node
     * @return nullptr if not applicable
     */
    virtual const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const { return this; }

    class AccessNodeExpand {
    public:
        virtual ~AccessNodeExpand() = default;

        virtual builder::StructuredSDFGBuilder& builder() = 0;

        /**
         * Creates a new sequence where the libNode was.
         * So if the original libNode was inside a block, after this call, there might be a lead-in block with nodes
         * that precede the libNode, the replacement sequence and a lead-out block with nodes that succeeded the
         * original libNode
         *
         * Nothing outside of this sequence must be modified, to keep the expansion stable and correct.
         *
         * @return new sequence
         */
        virtual structured_control_flow::Sequence& replace_with_sequence() = 0;

        enum class LoopType { For, Map };

        virtual structured_control_flow::StructuredLoop& replace_with_structured_loop(
            LoopType type,
            const symbolic::Symbol indvar,
            const symbolic::Condition condition,
            const symbolic::Expression init,
            const symbolic::Expression update,
            const ScheduleType& schedule_type
        ) = 0;

        /**
         * Creates an access node that represents the input of the original lib-node as accurately as possible,
         * including debug info and metadata Used as a scalar input (for pointers, the value of the pointer is read).
         * Only valid if the input was not requested as "Skip"
         * @param block block to insert into
         * @param input_idx the index in the list of access_dirs requested & input-connectors
         * @return newly created access node
         */
        virtual data_flow::AccessNode& add_scalar_input_access(structured_control_flow::Block& block, size_t input_idx) = 0;
        /**
         * Creates an access node that represents the input of the original lib-node as accurately as possible,
         * including debug info and metadata Used for indirect reads. Only valid if the original input is pointer-like
         * and it was requested for IndirectRead (or IndirectReadWrite)
         * @param block block to insert into
         * @param input_idx the index in the list of access_dirs requested & input-connectors
         * @return newly created access node
         */
        virtual data_flow::AccessNode&
        add_indirect_read_access(structured_control_flow::Block& block, size_t input_idx) = 0;
        /**
         * Creates an access node that represents the input of the original lib-node as accurately as possible,
         * including debug info and metadata Used for indirect writes. Only valid if the original input is pointer-like
         * and it was requested for IndirectWrite (or IndirectReadWrite)
         *
         * Example: the libnode gets a pointer to an array. The expansion will now write to elements of this array in a
         * loop
         * @param block block to insert into
         * @param input_idx the index in the list of access_dirs requested & input-connectors
         * @return newly created access node
         */
        virtual data_flow::AccessNode&
        add_indirect_write_access(structured_control_flow::Block& block, size_t input_idx) = 0;
        /**
         * Creates an access node that represents the output of the original lib-node as accurately as possible,
         * including debug info and metadata Used for outputs of the original lib-node.
         * @param block block to insert into
         * @param input_idx the index in the list of output-connectors
         * @return newly created access node
         */
        virtual data_flow::AccessNode& add_output_access(structured_control_flow::Block& block, size_t output_idx) = 0;

        /**
         * The expansion is now complete. Certain replacements that require understanding all of the changes may be
         * applied now
         */
        virtual ExpandOutcome successfully_expanded() = 0;
    };

    class ExpandContext {
    public:
        virtual ~ExpandContext() = default;

        /**
         * @brief start expansion with access nodes. The expander may cut the dataflow into input, and output.
         * If there is a return value, then the access nodes can be provided. Whether this required cutting up the
         * original dataflow or not does not matter
         *
         * Actual replacement of the operation must be done via one of the methods of AccessNodeExpand, such as
         * replace_with_sequence().
         *
         * @param access_dirs list matching the input_conns of the node. Indicating for each input, whether it will be
         * needed as read and/or write nodes ( because ptr-inputs may result in indirect reads or writes or both).
         * Non-ptr inputs may only be skipped or used as Scalar)
         * @return null, if cannot be provided. Otherwise a handle to create arbitrarily many copies of access nodes
         * fitting the access_dirs
         */
        virtual std::unique_ptr<AccessNodeExpand> replacement_requires_access_nodes(const std::vector<InputUse>&
                                                                                        access_dirs) = 0;

        // TODO virtual std::unique_ptr<DataflowExpand> replace_dataflow() = 0;
        // this would be a quicker way, when the expansion only consists of pure dataflow, where base expansion
        // infrastructure needs to migrate edges from the old node to the new nodes. Describing the new targets of edges
        // can be taken from `ElementWiseDataflowTensorNode.expand_operation_dataflow()`

        virtual ExpandOutcome unable() = 0;

        virtual ExpandOutcome unapplicable() = 0;
    };

    virtual ExpandOutcome handle_expand(
        ExpandContext& context, structured_control_flow::Block& block, data_flow::LibraryNode& node
    ) const = 0;
};

template<typename T>
class TypedLibNodeExpander : public LibNodeExpander {
public:
    using LibNodeExpander::handle_expand;

    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override {
        if (auto* cast = dynamic_cast<const T*>(&node)) {
            return this;
        }
        return nullptr;
    }

    virtual LibNodeExpander::ExpandOutcome
    handle_expand(LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, T& node) const = 0;

    LibNodeExpander::ExpandOutcome handle_expand(
        LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, data_flow::LibraryNode& node
    ) const override {
        auto& typed_node = static_cast<T&>(node);
        return handle_expand(context, block, typed_node);
    }
};

template<typename T>
class CodeLibNodeExpander : public TypedLibNodeExpander<T> {
    const data_flow::LibraryNodeCode code_;

public:
    CodeLibNodeExpander(const data_flow::LibraryNodeCode& code) : code_(code) {}

    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override {
        if (node.code() == code_) {
            return this;
        }
        return nullptr;
    }
};

} // namespace sdfg::passes
