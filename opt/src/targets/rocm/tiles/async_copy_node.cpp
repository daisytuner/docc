#include "sdfg/targets/rocm/tiles/async_copy_node.h"

namespace sdfg {
namespace rocm {
namespace tiles {

namespace {

// Preprocessor guard selecting the CDNA archs (gfx9xx) that support the
// asynchronous direct global->LDS load path (`global_load_lds` / vmcnt). RDNA
// (gfx10xx/11xx/12xx) lacks it, so the emitted #else keeps a synchronous copy.
constexpr const char* kCdnaArchGuard =
    "defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || "
    "defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)";

} // namespace

CpAsyncCopyNodeDispatcher::CpAsyncCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::CpAsyncCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CpAsyncCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::CpAsyncCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are addresses (reference memlets).
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const size_t bytes = node.bytes();

    const size_t words = bytes / 4;
    // CDNA has an asynchronous direct global->LDS load; RDNA does not.
    out.stream << "#if " << kCdnaArchGuard << std::endl;
    // CDNA: per-word async global->LDS load (in flight, tracked by vmcnt,
    // drained by the matching PipelineWait). dst points into __shared__.
    out.stream << "for (size_t __i = 0; __i < " << words << "; ++__i) "
               << "__builtin_amdgcn_global_load_lds(reinterpret_cast<const unsigned*>(" << src
               << ") + __i, reinterpret_cast<unsigned*>(" << dst << ") + __i, 4, 0, 0);" << std::endl;
    out.stream << "#else" << std::endl;
    // RDNA (and any non-CDNA): no async LDS path — copy synchronously.
    out.stream << "for (size_t __i = 0; __i < " << words << "; ++__i) "
               << "reinterpret_cast<float*>(" << dst << ")[__i] = reinterpret_cast<const float*>(" << src << ")[__i];"
               << std::endl;
    out.stream << "#endif" << std::endl;
}

VectorCopyNodeDispatcher::VectorCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::VectorCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void VectorCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::VectorCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are addresses (reference memlets).
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    // A width-matched integer vector: a raw byte move (no fp semantics), legal for
    // any element type as long as bytes is 4/8/16. int4/int2/int are builtin vector
    // types on both CUDA (vector_types.h) and HIP (hip_vector_types.h).
    const char* vec = node.bytes() == 16 ? "int4" : node.bytes() == 8 ? "int2" : "int";
    out.stream << "*reinterpret_cast<" << vec << "*>(" << dst << ") = *reinterpret_cast<const " << vec << "*>(" << src
               << ");" << std::endl;
}

PipelineCommitNodeDispatcher::PipelineCommitNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::PipelineCommitNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

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
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

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
