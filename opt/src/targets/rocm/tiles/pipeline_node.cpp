#include "sdfg/targets/rocm/tiles/pipeline_node.h"

namespace sdfg {
namespace rocm {
namespace tiles {

namespace {

// CDNA archs (gfx9xx) drain direct global->LDS loads on the flat vmcnt counter;
// RDNA lacks that path, so the emitted wait compiles out.
constexpr const char* kCdnaArchGuard =
    "defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || "
    "defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)";

} // namespace

PipelineCommitNodeDispatcher::PipelineCommitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::PipelineCommitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void PipelineCommitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    // No commit primitive on either arch: CDNA loads are tracked directly by
    // the hardware vmcnt (drained in PipelineWait); RDNA is synchronous.
}

PipelineWaitNodeDispatcher::PipelineWaitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::PipelineWaitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void PipelineWaitNodeDispatcher::
    dispatch(codegen::PrettyPrinter& stream, codegen::PrettyPrinter&, codegen::CodeSnippetFactory&) {
    const auto& node = static_cast<const ::sdfg::tiles::PipelineWaitNode&>(node_);
    // CDNA: drain direct-to-LDS loads on the flat vmcnt counter, keeping
    // keep_outstanding pipeline stages (each loads_per_group words) still in
    // flight. RDNA: synchronous, nothing outstanding to wait on.
    stream << "#if " << kCdnaArchGuard << std::endl;
    stream << "asm volatile(\"s_waitcnt vmcnt(" << (node.keep_outstanding() * node.loads_per_group()) << ")\");"
           << std::endl;
    stream << "#endif" << std::endl;
}

} // namespace tiles
} // namespace rocm
} // namespace sdfg
