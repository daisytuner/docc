/**
 * @file block_dispatcher.h
 * @brief Dispatchers for dataflow blocks and library nodes
 *
 * This file defines dispatchers for generating code from dataflow blocks and
 * library nodes. It includes:
 * - BlockDispatcher: Generates code for blocks containing dataflow
 * - DataFlowDispatcher: Generates code for dataflow graphs
 * - LibraryNodeDispatcher: Base class for library node code generation
 *
 * ## Library Node Dispatchers
 *
 * LibraryNodeDispatcher is the base class for generating code from library nodes.
 * Each library node type can have a custom dispatcher registered in the
 * LibraryNodeDispatcherRegistry. The dispatcher is responsible for:
 * - Generating the appropriate library call or inline code
 * - Handling input/output data access
 * - Managing implementation-specific details
 *
 * Dispatchers work together with the library node's implementation_type:
 * - If implementation_type is NONE, the node may be expanded first
 * - If implementation_type specifies a library (e.g., BLAS), the dispatcher
 *   generates a call to that library
 *
 * @see node_dispatcher_registry.h for dispatcher registration
 * @see data_flow::LibraryNode for library node definition
 */

#pragma once

#include "sdfg/codegen/dispatchers/node_dispatcher.h"
#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace codegen {

class BlockDispatcher : public NodeDispatcher {
private:
    const structured_control_flow::Block& node_;

public:
    BlockDispatcher(
        LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& node,
        InstrumentationPlan& instrumentation_plan,
        ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        PrettyPrinter& main_stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override;
};

class DataFlowDispatcher {
private:
    LanguageExtension& language_extension_;
    const Function& function_;
    const data_flow::DataFlowGraph& data_flow_graph_;
    const InstrumentationPlan& instrumentation_plan_;

    void dispatch_ref(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_deref_src(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_deref_dst(PrettyPrinter& stream, const data_flow::Memlet& memlet);

    void dispatch_tasklet(PrettyPrinter& stream, const data_flow::Tasklet& tasklet);

    void dispatch_library_node(
        PrettyPrinter& stream,
        PrettyPrinter& globals_stream,
        CodeSnippetFactory& library_snippet_factory,
        const data_flow::LibraryNode& libnode
    );

public:
    DataFlowDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const InstrumentationPlan& instrumentation_plan
    );

    void dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);
};

struct CodegenOutput {
    PrettyPrinter& stream;
    PrettyPrinter& globals_stream;
    CodeSnippetFactory& library_snippet_factory;
    LanguageExtension& language_extension;
};

/**
 * Short-lived container for an input into a dispatcher (based on dflow edges)
 * references are only valid up until the return of the dispatch call.
 */
struct DispatchInput {
    const std::string expr;
    const data_flow::Memlet& edge;
    bool is_locally_modifiable;
};

struct DispatchOutput {
    const std::string* local_name;
    std::unique_ptr<types::IType> out_type;
    bool used;
};

/**
 * @class LibraryNodeDispatcher
 * @brief Base class for library node code generation dispatchers
 *
 * LibraryNodeDispatcher provides the interface for generating code from library
 * nodes. Subclasses implement specific code generation for different library
 * operations (BLAS calls, memory operations, etc.).
 *
 * A modern impl overrides [dispatch_code_with_edges].
 *
 * CodegenOutput is used as a wrapper around multiple args needed. This makes extending the args provided a simpler
 * code-change. We expect this to also include state to handle data-flow edges in the future.
 *
 * The base class collects all inputs and outputs and provides them. For inputs, this is 1:1 with edges, since outputs
 * in dataflow can have multiple edges, no edge is attached to them. The dispatcher only needs to provide the output per
 * connector, the base class will handle providing the output to all edges as needed.
 *
 * Legacy code can still override the [dispatch_code] as before and is provided with almost the same environment of
 * variables.
 *
 * Note the class handles ptrs correctly, meaning there is no need to consider inputs and outputs overlapping in any
 * way. If a pointer is provided as input, that can be used to change all the data it points to. A pointer output is
 * only needed if a pointer is selected, created or modified. Not if the data behind is changed.
 */
class LibraryNodeDispatcher {
protected:
    LanguageExtension& language_extension_; ///< Language extension for code generation
    const Function& function_; ///< Function context
    const data_flow::DataFlowGraph& data_flow_graph_; ///< Containing dataflow graph
    const data_flow::LibraryNode& node_; ///< Library node being dispatched

public:
    /**
     * @brief Construct a library node dispatcher
     * @param language_extension Language extension for code generation
     * @param function Function context
     * @param data_flow_graph Containing dataflow graph
     * @param node Library node to dispatch
     */
    LibraryNodeDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node
    );

    virtual ~LibraryNodeDispatcher() = default;

    /**
     * @brief Begin code generation for the node
     * @param stream Output stream for generated code
     * @return True if a declaration was generated
     */
    virtual bool begin_node(PrettyPrinter& stream) { return false; }

    /**
     * @brief End code generation for the node
     * @param stream Output stream for generated code
     * @param has_declaration Whether a declaration was generated
     */
    virtual void end_node(PrettyPrinter& stream, bool has_declaration) {}

    /**
     * @brief Dispatch the library node to code
     * @param stream Main code stream
     * @param globals_stream Global declarations stream
     * @param library_snippet_factory Factory for library code snippets
     */
    virtual void
    dispatch(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory);

    /**
     * @brief Generate the operation-specific code
     *
     * Subclasses override this method to generate the actual library call or
     * inline implementation. This might include:
     * - BLAS library calls (cblas_dgemm, etc.)
     * - Standard library calls (memcpy, malloc, etc.)
     * - Custom inline implementations
     *
     * @param stream Main code stream
     * @param globals_stream Global declarations stream
     * @param library_snippet_factory Factory for library code snippets
     */
    virtual void
    dispatch_code(PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory) {}

    virtual void dispatch_code_with_edges(
        CodegenOutput& out, std::vector<DispatchInput>& inputs, std::vector<DispatchOutput>& outputs
    );

    /**
     * @brief Get instrumentation information for this node
     * @return Instrumentation information
     */
    virtual InstrumentationInfo instrumentation_info() const;

protected:
    /**
     * Ensure that [input] points to a local copy that can be changed without modifying the actual source of the input
     * Should only be needed to ensure that legacy [dispatch_code] code can act as before, where it always was a copy.
     */
    void require_locally_modifiable_var(CodegenOutput& out, DispatchInput& input) const;

    /**
     * Declare a variable of the type of the output and with the name "var_name".
     * Compatible to preexisting [dispatch_code] impls.
     */
    void pre_allocate_output(CodegenOutput& out, DispatchOutput& output, const std::string& var_name) const;

    /**
     * The dispatcher as created an output that is identified by [result_identifier].
     * Use that as the source for the output edges of [output]
     */
    void register_output(DispatchOutput& output, const std::string& result_identifier) const;

    /**
     * This is a temporary & hopefully short-lived workaround. For a sensible and flexible data flow graph you want to
     * know the available data (by some unique ID, like the vertex IDs) And then just use those whenever they are
     * declared as an input. Edges can possibly apply some on the fly transformations, like selects or casts. Access
     * Nodes / whatever new modeling of memory writes & reads we then have, needs to dispatch whatever side effects they
     * need themselves, just like lib-nodes do. This saves unneeded copying around which can cause issues and
     * inadvertently remove or add side effects. Also makes the code redundant.
     *
     * But this also requires a per-dataflow-graph state of data-handles-by-id. But this would also be what you would
     * want to emit LLVM-IR directly
     */
    void copy_output(CodegenOutput& out, const DispatchOutput& output, const data_flow::Memlet& oedge) const;
};

} // namespace codegen
} // namespace sdfg
