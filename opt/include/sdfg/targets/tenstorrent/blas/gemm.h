#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/tenstorrent/library_node_dispatcher.h"

namespace sdfg::tenstorrent::blas {

class GEMMNodeDispatcher_Tenstorrent : public LibraryNodeDispatcherBase<math::blas::GEMMNode> {
public:
    GEMMNodeDispatcher_Tenstorrent(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::GEMMNode& node
    );

    std::pair<symbolic::Expression, symbolic::Expression> emit_padded_size(
        codegen::PrettyPrinter& stream, const std::string& var_name, const symbolic::Expression size, int pad_to_mul
    ) const;

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;
};


} // namespace sdfg::tenstorrent::blas
