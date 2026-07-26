/**
 * @file node_dispatcher.h
 * @brief Dispatcher base class for code generation from control flow nodes
 *
 * Dispatchers are responsible for generating code from SDFG control flow nodes.
 * Each control flow node type (Block, Map, For, While, etc.) has a corresponding
 * dispatcher that knows how to generate the appropriate code.
 *
 * The dispatcher pattern allows separation of code generation logic from the
 * SDFG representation, making it easier to support multiple target languages
 * and platforms.
 *
 * @see node_dispatcher_registry.h for dispatcher registration
 * @see block_dispatcher.h for dataflow and library node dispatching
 */

#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace codegen {

/**
 * @class NodeDispatcher
 * @brief Base class for dispatching control flow nodes to code
 *
 * NodeDispatcher provides the interface for generating code from SDFG control
 * flow nodes. Each node type has a corresponding dispatcher implementation
 * that knows how to generate appropriate code for that node.
 */
class NodeDispatcher {
private:
    structured_control_flow::ControlFlowNode& node_;

    // Teardown state captured during dispatch() so that end_dispatch() can be replayed early.
    bool begin_node_applied_ = false;
    bool end_dispatched_ = false;

protected:
    LanguageExtension& language_extension_;

    StructuredSDFG& sdfg_;

    InstrumentationPlan& instrumentation_plan_;

    ArgCapturePlan& arg_capture_plan_;

    analysis::AnalysisManager& analysis_manager_;

    virtual bool begin_node(PrettyPrinter& stream);

    virtual void end_node(PrettyPrinter& stream, bool has_declaration);

    /**
     * Emits the closing code of this node (end instrumentation, arg capture teardown, end_node).
     *
     * This is called automatically by dispatch() after dispatch_node(). Nodes that emit a
     * control-flow terminator (e.g. Return) must call this manually *before* emitting the
     * terminating statement, otherwise the teardown code would be unreachable at runtime.
     *
     * The method is idempotent: subsequent calls (including the automatic one from dispatch())
     * are no-ops once the teardown has been emitted.
     */
    void end_dispatch(PrettyPrinter& main_stream);

    /**
     * Bad design. We already have fields in this class and have to bind an instance to a single node.
     * Just set a field to sth. of the instrumentation plan where the node can register more details. But this is a
     * breaking change
     */
    virtual InstrumentationInfo instrumentation_info() const;

public:
    NodeDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    virtual ~NodeDispatcher() = default;

    virtual void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) = 0;

    /**
     * Emits the end-instrumentation and arg-capture teardown for this node *without* the
     * structural end_node() and without marking the node as finalized.
     *
     * This is used to tear down enclosing instrumented regions before a control-flow
     * terminator (e.g. return) leaves them: the terminating node walks its ancestor chain
     * and calls this on each enclosing region so their instrumentation exit/finalize calls
     * are reachable. The regions' normal-position teardown remains in place for control-flow
     * paths that do not hit the terminator.
     */
    void emit_instrumentation_exit(PrettyPrinter& main_stream);

    virtual void
    dispatch(PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
};

} // namespace codegen
} // namespace sdfg
