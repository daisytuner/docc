#pragma once

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/data_flow/library_nodes/math/blas/dot_node.h"
#include "sdfg/targets/tenstorrent/library_node_dispatcher.h"

namespace sdfg::tenstorrent::blas {

class DotNodeDispatcher_Tenstorrent : public LibraryNodeDispatcherBase<math::blas::DotNode> {
public:
    DotNodeDispatcher_Tenstorrent(
        codegen::LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const math::blas::DotNode& node
    );

    static constexpr bool force_close_device = false;

    std::pair<symbolic::Expression, symbolic::Expression> emit_padded_size(
        codegen::PrettyPrinter& stream, const std::string& var_name, const symbolic::Expression size, int pad_to_mul
    ) const;

    void dispatch(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};

} // namespace sdfg::tenstorrent::blas
