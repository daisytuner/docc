#include "sdfg/targets/rocm/tiles/tile_copy_node.h"

namespace sdfg {
namespace rocm {
namespace tiles {

namespace {

// CDNA archs (gfx9xx) with the asynchronous direct global->LDS load path
// (`global_load_lds` / vmcnt). RDNA lacks it, so the #else keeps a synchronous copy.
constexpr const char* kCdnaArchGuard =
    "defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || "
    "defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)";

} // namespace

TileCopyNodeDispatcher::TileCopyNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const ::sdfg::tiles::TileCopyNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {
}

void TileCopyNodeDispatcher::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    const auto& node = static_cast<const ::sdfg::tiles::TileCopyNode&>(node_);
    // Connector order {"_dst", "_src"} — both are the bare base pointers.
    const std::string& dst = inputs.at(0).expr;
    const std::string& src = inputs.at(1).expr;
    const auto pt = inputs.at(0).edge.base_type().primitive_type();

    // CpAsync on CDNA is a per-word direct global->LDS load (tracked by vmcnt, drained
    // by the pipeline wait); RDNA has no async path, so #else copies synchronously.
    auto async_stmt = [](const std::string& dst_addr, const std::string& src_addr, size_t bytes) {
        // global_load_lds moves 1/2/4 bytes per lane per call; chunk at the largest
        // power-of-2 <= 4 that divides `bytes` so sub-dword (e.g. fp16 = 2B) elements
        // still copy (a hardcoded 4B dword gave bytes/4 == 0 and a dead loop for fp16).
        size_t chunk = 4;
        while (chunk > 1 && bytes % chunk != 0) {
            chunk /= 2;
        }
        const std::string words = std::to_string(bytes / chunk);
        const std::string sz = std::to_string(chunk);
        const char* word = chunk == 4 ? "unsigned" : chunk == 2 ? "unsigned short" : "unsigned char";
        std::string s;
        s += "#if ";
        s += kCdnaArchGuard;
        s += "\n";
        s += "for (size_t __i = 0; __i < " + words + "; ++__i) __builtin_amdgcn_global_load_lds(" +
             "reinterpret_cast<const " + word + "*>(" + src_addr + ") + __i, reinterpret_cast<" + word + "*>(" +
             dst_addr + ") + __i, " + sz + ", 0, 0);\n";
        s += "#else\n";
        s += "for (size_t __i = 0; __i < " + words + "; ++__i) reinterpret_cast<" + word + "*>(" + dst_addr +
             ")[__i] = reinterpret_cast<const " + word + "*>(" + src_addr + ")[__i];\n";
        s += "#endif";
        return s;
    };
    ::sdfg::tiles::emit_cooperative_copy_loop(language_extension_, out.stream, node, dst, src, pt, async_stmt);
}

} // namespace tiles
} // namespace rocm
} // namespace sdfg
