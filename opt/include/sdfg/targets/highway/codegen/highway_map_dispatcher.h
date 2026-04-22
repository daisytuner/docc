#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/codegen/dispatchers/node_dispatcher.h>
#include <sdfg/codegen/instrumentation/instrumentation_info.h>
#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/tasklet.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>

namespace sdfg {
namespace highway {

constexpr std::string HIGHWAY_SNIPPET_EXT = "highway.cpp";

class HighwayMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

    std::vector<std::string> arguments_;
    std::vector<std::string> arguments_declaration_;
    types::PrimitiveType vec_type_;

    std::set<std::string> arguments_lookup_;
    std::set<std::string> locals_;
    symbolic::SymbolSet local_symbols_;
    symbolic::Symbol indvar_;

public:
    HighwayMapDispatcher(
        codegen::LanguageExtension& language_extension,
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Map& node,
        codegen::InstrumentationPlan& instrumentation_plan,
        codegen::ArgCapturePlan& arg_capture_plan
    );

    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    std::string declaration(const std::string& container, const types::Scalar& type);

    void dispatch_iedge(codegen::PrettyPrinter& library_stream, const data_flow::Memlet& memlet);

    void dispatch_highway(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& library_stream,
        structured_control_flow::ControlFlowNode& node
    );

    void dispatch_highway(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& library_stream,
        structured_control_flow::Sequence& node
    );

    void dispatch_highway(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& library_stream,
        structured_control_flow::Block& node
    );

    void dispatch_highway(
        codegen::CodeSnippetFactory& library_snippet_factory,
        codegen::PrettyPrinter& library_stream,
        data_flow::CodeNode& node
    );

    void dispatch_kernel_call(codegen::PrettyPrinter& main_stream, const std::string& kernel_name);

    void dispatch_kernel_declaration(codegen::PrettyPrinter& globals_stream, const std::string& kernel_name);

    void dispatch_kernel_preamble(codegen::PrettyPrinter& library_stream, const std::string& kernel_name);

    void dispatch_kernel_body(codegen::CodeSnippetFactory& library_snippet_factory, codegen::PrettyPrinter& library_stream);

    codegen::InstrumentationInfo instrumentation_info() const override;

    static std::string daisy_vec(const types::PrimitiveType& type);

    static std::string tasklet(data_flow::Tasklet& tasklet);

    static std::string cmath_node(math::cmath::CMathNode& cmath_node);
};

} // namespace highway
} // namespace sdfg
