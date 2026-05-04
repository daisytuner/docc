#pragma once

#include <concepts>
#include <string>
#include <unordered_map>

#include "sdfg/analysis/base_user_visitor.h"
#include "sdfg/function.h"

namespace sdfg::analysis {
/**
 * Concept for policies that handle pointer escape and overwrite events.
 *
 * An EscapePolicy receives notifications when a pointer-typed container
 * is observed to escape (its value becomes visible outside the current scope)
 * or to be overwritten (a new value is assigned to the pointer variable itself).
 */
template<typename P>
concept EscapePolicy = requires(P& p, const std::string& container, const ControlFlowNode* node, const Element* user) {
    p.on_escape(container, node, user);
};

/**
 * Reusable BaseUserAnalyzer that detects pointer escapes and overwrites.
 *
 * For each of the 5 BaseUserAnalyzer callbacks, it checks whether the access
 * involves a pointer-typed container and whether the operation constitutes
 * an escape (value leaks out of our control) or an overwrite (the pointer
 * variable itself is reassigned).  Detected events are forwarded to the
 * policy object via static dispatch.
 *
 * Does NOT walk the SDFG itself — must be driven by a BaseUserVisitor
 * (or future CompositeUserVisitor) that calls the callbacks.
 */
template<EscapePolicy Policy>
class PointerEscapeAnalyzer : public virtual BaseUserAnalyzer {
    const StructuredSDFG& sdfg_;
    Policy& policy_;

public:
    PointerEscapeAnalyzer(const StructuredSDFG& sdfg, Policy& policy) : sdfg_(sdfg), policy_(policy) {}

    void use_as_return_src(const std::string& container, const Return& ret) override {
        if (sdfg_.type(container).type_id() == types::TypeID::Pointer) {
            policy_.on_escape(container, &ret, &ret);
        }
    }

    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {
        auto& type = sdfg_.type(container);
        if (type.type_id() == types::TypeID::Pointer) {
            policy_.on_escape(container, node, user);
        }
    }

    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        auto& type = sdfg_.type(container);
        if (edge.is_src_pointed_to_address_leak(type) || edge.is_src_address_leak()) {
            // pulls a reference to the owned memory area or can alias the entire pointer

            policy_.on_escape(container, &block, &edge);
            // it may not be, but this is the safest
            // assumption. other passes can forward the original container and fold it into accesses
        }
    }

    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {}
    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {}
};

template<typename P>
concept OverwritePolicy =
    requires(P& p, const std::string& container, const ControlFlowNode* node, const Element* user) {
        p.on_overwrite(container, node, user);
    };

template<OverwritePolicy Policy>
class PointerOverwriteAnalyzer : public BaseUserAnalyzer {
    const StructuredSDFG& sdfg_;
    Policy& policy_;

public:
    PointerOverwriteAnalyzer(const StructuredSDFG& sdfg, Policy& policy) : sdfg_(sdfg), policy_(policy) {}

    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        if (edge.is_dst_write()) { // writes to the ptr
            auto& type = sdfg_.type(container);
            if (type.type_id() == types::TypeID::Pointer) {
                policy_.on_overwrite(container, &block, &edge);
            }
        }
    }

    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {
        auto name = container->get_name();
        if (sdfg_.type(name).type_id() == types::TypeID::Pointer) {
            policy_.on_overwrite(name, node, user);
        }
    }

    void use_as_return_src(const std::string& container, const Return& ret) override {}
    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {}
    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {}
};

template<typename P>
concept PointerUsedPolicy =
    requires(P& p, const std::string& container, const ControlFlowNode* node, const data_flow::Memlet* user) {
        p.on_write_via(container, node, user);
        p.on_read_via(container, node, user);
    };

/**
 * Indirect access via pointer happened
 **/
template<PointerUsedPolicy Policy>
class PointerUsedAnalyzer : public BaseUserAnalyzer {
    const StructuredSDFG& sdfg_;
    Policy& policy_;

public:
    PointerUsedAnalyzer(const StructuredSDFG& sdfg, Policy& policy) : sdfg_(sdfg), policy_(policy) {}

    void use_as_src_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        if (edge.is_src_pointed_to_read()) {
            policy_.on_read_via(container, &block, &edge);
        }
    }

    void use_as_dst_node(
        const std::string& container,
        const data_flow::AccessNode& node,
        const data_flow::Memlet& edge,
        const Block& block
    ) override {
        if (edge.is_dst_pointed_to_write()) {
            policy_.on_write_via(container, &block, &edge);
        }
    }

    void use_as_symbol_write(
        const symbolic::Symbol& container, const ControlFlowNode* node, const Element* user, SymbolWriteLocation loc
    ) override {}

    void use_as_return_src(const std::string& container, const Return& ret) override {}

    void use_as_symbol_read(
        const std::string& container,
        const ControlFlowNode* node,
        const Element* user,
        SymbolReadLocation loc,
        int loc_index,
        symbolic::Expression expr
    ) override {}
};


} // namespace sdfg::analysis
