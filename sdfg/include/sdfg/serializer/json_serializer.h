#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/types/type.h"
#include "symengine/logic.h"
#include "symengine/printers/codegen.h"

namespace sdfg {
namespace serializer {

class JSONSerializer {
private:
    bool recurse_;

    // Sub-type of structured loops
    void json_to_map_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );

    void json_to_reduce_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );

public:
    JSONSerializer(bool recurse = true) : recurse_(recurse) {}

    virtual nlohmann::json serialize(
        const sdfg::StructuredSDFG& sdfg,
        analysis::AnalysisManager* analysis_manager = nullptr,
        structured_control_flow::Sequence* root = nullptr
    );

    void serialize_node(nlohmann::json& j, const structured_control_flow::ControlFlowNode& node);

    std::unique_ptr<sdfg::StructuredSDFG> deserialize(nlohmann::json& j);

    void structure_definition_to_json(nlohmann::json& j, const sdfg::types::StructureDefinition& definition);
    void type_to_json(nlohmann::json& j, const sdfg::types::IType& type);
    void dataflow_to_json(nlohmann::json& j, const sdfg::data_flow::DataFlowGraph& dataflow);

    void sequence_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Sequence& sequence);
    void block_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Block& block);
    void if_else_to_json(nlohmann::json& j, const sdfg::structured_control_flow::IfElse& if_else_node);
    void while_node_to_json(nlohmann::json& j, const sdfg::structured_control_flow::While& while_node);
    void break_node_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Break& break_node);
    void continue_node_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Continue& continue_node);
    void return_node_to_json(nlohmann::json& j, const sdfg::structured_control_flow::Return& return_node);

    // Includes all sub-types of StructuredLoop (for, map, reduce)
    void structured_loop_to_json(nlohmann::json& j, const sdfg::structured_control_flow::StructuredLoop& loop_node);

    void debug_info_to_json(nlohmann::json& j, const sdfg::DebugInfo& debug_info);

    void schedule_type_to_json(nlohmann::json& j, const sdfg::structured_control_flow::ScheduleType& schedule_type);

    void storage_type_to_json(nlohmann::json& j, const sdfg::types::StorageType& storage_type);

    void json_to_structure_definition(const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder);
    void json_to_dataflow(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    );

    void json_to_sequence(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& sequence
    );
    void json_to_block_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    void json_to_if_else_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    void json_to_while_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    void json_to_break_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    void json_to_continue_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    void json_to_return_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    // Includes all sub-types of StructuredLoop (for, map, reduce)
    void json_to_structured_loop_node(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Sequence& parent,
        control_flow::Assignments& assignments
    );
    std::unique_ptr<sdfg::types::IType> json_to_type(const nlohmann::json& j);
    std::vector<std::pair<std::string, types::Scalar>> json_to_arguments(const nlohmann::json& j);
    DebugInfo json_to_debug_info(const nlohmann::json& j);

    ScheduleType json_to_schedule_type(const nlohmann::json& j);

    types::StorageType json_to_storage_type(const nlohmann::json& j);

    static std::string expression(const symbolic::Expression expr);

    symbolic::Expression json_to_expr(const nlohmann::json& j);

    static void writeToFile(const StructuredSDFG& sdfg, const std::filesystem::path& file);
};

class JSONSymbolicPrinter : public SymEngine::BaseVisitor<JSONSymbolicPrinter, SymEngine::CodePrinter> {
public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    // Logical expressions
    void bvisit(const SymEngine::Equality& x);
    void bvisit(const SymEngine::Unequality& x);

    void bvisit(const SymEngine::LessThan& x);
    void bvisit(const SymEngine::StrictLessThan& x);

    // Min and Max
    void bvisit(const SymEngine::Min& x);
    void bvisit(const SymEngine::Max& x);
};

/**
 * @class LibraryNodeSerializer
 * @brief Base class for library node serialization
 *
 * LibraryNodeSerializer provides the interface for serializing and deserializing
 * library nodes to/from JSON format. Each library node type should provide a
 * custom serializer that knows how to handle its specific data members.
 *
 * Serializers must handle:
 * - All node-specific parameters (shapes, axes, options, etc.)
 * - Debug information
 * - Element IDs
 * - Operation codes
 *
 * The serialization format is JSON-based and supports round-trip serialization
 * (serialize then deserialize produces an equivalent node).
 */
class LibraryNodeSerializer {
public:
    virtual ~LibraryNodeSerializer() = default;

    /**
     * @brief Serialize a library node to JSON
     * @param library_node Library node to serialize
     * @return JSON representation of the node
     */
    virtual nlohmann::json serialize(const sdfg::data_flow::LibraryNode& library_node) = 0;

    /**
     * @brief Deserialize a library node from JSON
     * @param j JSON object containing serialized node data
     * @param builder SDFG builder for creating the node
     * @param parent Parent block for the node
     * @return Reference to the deserialized library node
     */
    virtual data_flow::LibraryNode& deserialize(
        const nlohmann::json& j,
        sdfg::builder::StructuredSDFGBuilder& builder,
        sdfg::structured_control_flow::Block& parent
    ) = 0;
};

using LibraryNodeSerializerFn = std::function<std::unique_ptr<LibraryNodeSerializer>()>;

/**
 * @class LibraryNodeSerializerRegistry
 * @brief Registry for library node serializers
 *
 * This registry maps library node codes (operation identifiers) to serializer
 * factory functions. Each library node type registers its serializer at program
 * startup, allowing the JSON serializer to correctly handle all node types.
 *
 * The registry is thread-safe and follows the singleton pattern.
 *
 * ## Usage
 *
 * Register a serializer:
 * @code
 * LibraryNodeSerializerRegistry::instance().register_library_node_serializer(
 *     "Add",
 *     []() { return std::make_unique<ElementWiseBinaryNodeSerializer<AddNode>>(); }
 * );
 * @endcode
 *
 * Lookup a serializer:
 * @code
 * auto factory = LibraryNodeSerializerRegistry::instance()
 *                    .get_library_node_serializer("Add");
 * if (factory) {
 *     auto serializer = factory();
 *     json j = serializer->serialize(node);
 * }
 * @endcode
 */
class LibraryNodeSerializerRegistry {
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LibraryNodeSerializerFn> factory_map_;

public:
    static LibraryNodeSerializerRegistry& instance() {
        static LibraryNodeSerializerRegistry registry;
        return registry;
    }

    void register_library_node_serializer(std::string library_node_code, LibraryNodeSerializerFn fn);

    LibraryNodeSerializerFn get_library_node_serializer(std::string library_node_code);

    size_t size() const;
};

void register_default_serializers();

} // namespace serializer
} // namespace sdfg
