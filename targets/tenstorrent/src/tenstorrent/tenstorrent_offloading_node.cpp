#include "docc/target/tenstorrent/tenstorrent_offloading_node.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "../../../../sdfg/include/sdfg/targets/offloading/data_offloading_node.h"
#include "docc/target/tenstorrent/codegen.h"
#include "docc/target/tenstorrent/plugin.h"
#include "docc/target/tenstorrent/tenstorrent_transfer_arg.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"
#include "sdfg/graph/graph.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace tenstorrent {

TTDataOffloadingNode::TTDataOffloadingNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    bool blocking,
    std::string device_handle,
    symbolic::Expression size,
    symbolic::Expression page_size,
    offloading::DataTransferDirection transfer_direction,
    offloading::BufferLifecycle allocation_handling,
    int cq_no
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Tenstorrent_Offloading,
          {},
          {},
          transfer_direction,
          allocation_handling,
          size
      ),
      blocking_(blocking), page_size_(page_size), cq_no_(cq_no), device_handle_(device_handle) {
    if (!is_NONE(transfer_direction)) {
        this->inputs_.push_back("_src");
        this->outputs_.push_back("_dst");
    } else if (is_ALLOC(allocation_handling)) {
        this->outputs_.push_back("_ret");
    } else if (is_FREE(allocation_handling)) {
        this->inputs_.push_back("_ptr");
        this->outputs_.push_back("_ptr");
    }
}

void TTDataOffloadingNode::validate(const Function& function) const {
    // Prevent copy-in and free
    if (this->is_h2d() && this->is_free()) {
        throw InvalidSDFGException("TTDataOffloadingNode: Combination copy-in and free is not allowed");
    }

    // Prevent copy-out and alloc
    if (this->is_d2h() && this->is_alloc()) {
        throw InvalidSDFGException("TTDataOffloadingNode: Combination copy-out and alloc is not allowed");
    }
}

bool TTDataOffloadingNode::blocking() const { return this->blocking_; }

const symbolic::Expression TTDataOffloadingNode::page_size() const { return this->page_size_; }

int TTDataOffloadingNode::cq_no() const { return this->cq_no_; }

std::string TTDataOffloadingNode::device_handle() const { return this->device_handle_; }

const symbolic::Expression TTDataOffloadingNode::alloc_size() const {
    auto size = this->size();
    auto page_size = this->page_size();
    return TransferArg::calc_allocated_size(size, page_size);
}

std::unique_ptr<data_flow::DataFlowNode> TTDataOffloadingNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<TTDataOffloadingNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->blocking(),
        this->device_handle(),
        this->size(),
        this->page_size(),
        this->transfer_direction(),
        this->buffer_lifecycle(),
        this->cq_no()
    );
}

symbolic::SymbolSet TTDataOffloadingNode::symbols() const {
    if (this->page_size().is_null()) {
        return offloading::DataOffloadingNode::symbols();
    }
    auto symbols = offloading::DataOffloadingNode::symbols();
    auto page_size_atoms = symbolic::atoms(this->page_size());
    symbols.insert(page_size_atoms.begin(), page_size_atoms.end());
    return symbols;
}

void TTDataOffloadingNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    offloading::DataOffloadingNode::replace(old_expression, new_expression);
    this->page_size_ = symbolic::subs(this->page_size_, old_expression, new_expression);
}

bool TTDataOffloadingNode::redundant_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::redundant_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const TTDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->page_size(), other_node.page_size())) {
        return false;
    }
    if (this->cq_no() != other_node.cq_no()) {
        return false;
    }
    if (this->device_handle() != other_node.device_handle()) {
        return false;
    }

    return true;
}

bool TTDataOffloadingNode::equal_with(const offloading::DataOffloadingNode& other) const {
    if (!offloading::DataOffloadingNode::equal_with(other)) {
        return false;
    }

    auto& other_node = static_cast<const TTDataOffloadingNode&>(other);
    if (!symbolic::null_safe_eq(this->page_size(), other_node.page_size())) {
        return false;
    }
    if (this->cq_no() != other_node.cq_no()) {
        return false;
    }
    if (this->device_handle() != other_node.device_handle()) {
        return false;
    }

    return true;
}

bool TTDataOffloadingNode::is_same_target(const offloading::DataOffloadingNode& other) const {
    auto* other_tt = dynamic_cast<const TTDataOffloadingNode*>(&other);
    return other_tt != nullptr && other_tt->device_handle() == this->device_handle();
}

TTDataOffloadingNodeDispatcher::TTDataOffloadingNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const data_flow::LibraryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void TTDataOffloadingNodeDispatcher::dispatch_enqueue_read_safe(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const std::string& dev_handle,
    symbolic::Expression size,
    bool blocking,
    int cq_no
) {
    stream << "__daisy_tt_d2h_transfer(" << std::endl;
    stream << "\t" << dev_handle << ",";
    stream << "\t" << cq_no << "," << std::endl;
    stream << "\t" << src_ptr << ", " << dst_ptr << ", " << std::endl;
    stream << "\t" << language_extension.expression(std::move(size)) << "," << std::endl;
    stream << "\t" << blocking << std::endl;
    stream << ");" << std::endl;

    emit_d2h_transfer_helper_once(language_extension, globals_stream, library_snippet_factory);
}

void TTDataOffloadingNodeDispatcher::dispatch_enqueue_read_full(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const std::string& dev_handle,
    bool blocking,
    int cq_no
) {
    stream << "tt::tt_metal::EnqueueReadBuffer(" << std::endl;
    stream << "\t" << dev_handle << "->command_queue(" << cq_no << "), " << std::endl;
    stream << "\t" << src_ptr << ", " << dst_ptr << ", " << std::endl;
    stream << "\t" << blocking << std::endl;
    stream << ");" << std::endl;
}

void TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_safe(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const std::string& dev_handle,
    symbolic::Expression size,
    bool blocking,
    int cq_no
) {
    stream << "__daisy_tt_h2d_transfer(" << std::endl;
    stream << "\t" << dev_handle << ", " << std::endl;
    stream << "\t" << cq_no << ", " << std::endl;
    stream << "\t" << dst_ptr << ", " << src_ptr << ", " << std::endl;
    stream << "\t" << language_extension.expression(std::move(size)) << "," << std::endl;
    stream << "\t" << blocking << std::endl;
    stream << ");" << std::endl;

    emit_h2d_transfer_helper_once(language_extension, globals_stream, library_snippet_factory);
}

void TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_full(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const std::string& src_ptr,
    const std::string& dst_ptr,
    const std::string& dev_handle,
    bool blocking,
    int cq_no
) {
    stream << "tt::tt_metal::EnqueueWriteBuffer(" << std::endl;
    stream << "\t" << dev_handle << "->command_queue(" << cq_no << "), " << std::endl;
    stream << "\t" << dst_ptr << ", " << src_ptr << ", " << std::endl;
    stream << "\t" << blocking << std::endl;
    stream << ");" << std::endl;
}

void TTDataOffloadingNodeDispatcher::dispatch_allocate(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::LanguageExtension& language_extension,
    const std::string& var_name,
    const std::string& device_handle,
    const symbolic::Expression size,
    const symbolic::Expression page_size
) {
    stream << var_name << " = tt::tt_metal::CreateBuffer({" << std::endl;
    auto orgIndent = stream.indent();
    stream.setIndent(orgIndent + 4);
    stream << ".device = " << device_handle << ", " << std::endl;
    stream << ".size = static_cast<tt::tt_metal::DeviceAddr>(" << language_extension.expression(size) << "),"
           << std::endl;
    stream << ".page_size = static_cast<tt::tt_metal::DeviceAddr>(" << language_extension.expression(page_size) << "), "
           << std::endl;
    stream << ".buffer_type = tt::tt_metal::BufferType::DRAM" << std::endl;
    stream.setIndent(orgIndent);
    stream << "});" << std::endl;
}

void TTDataOffloadingNodeDispatcher::dispatch(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& offloading_node = static_cast<const TTDataOffloadingNode&>(node_);

    auto& graph = this->node_.get_parent();
    auto& oedge = *graph.out_edges((this->node_)).begin();

    auto& dst = dynamic_cast<const data_flow::AccessNode&>(oedge.dst());
    std::string dst_name = this->language_extension_.access_node(dst);

    if (offloading_node.is_alloc()) {
        auto& org_size = offloading_node.alloc_size();
        auto& page_size = offloading_node.page_size();

        TTDataOffloadingNodeDispatcher::dispatch_allocate(
            stream,
            globals_stream,
            library_snippet_factory,
            language_extension_,
            dst_name,
            offloading_node.device_handle(),
            org_size,
            page_size
        );
    }

    if (offloading_node.has_transfer()) {
        auto& iedge = *graph.in_edges((this->node_)).begin();

        auto& src = dynamic_cast<const data_flow::AccessNode&>(iedge.src());
        std::string src_name = this->language_extension_.access_node(src);

        if (offloading_node.is_d2h()) {
            TTDataOffloadingNodeDispatcher::dispatch_enqueue_read_safe(
                language_extension_,
                stream,
                globals_stream,
                library_snippet_factory,
                src_name,
                dst_name,
                offloading_node.device_handle(),
                offloading_node.size(),
                offloading_node.blocking(),
                offloading_node.cq_no()
            );
        } else if (offloading_node.is_h2d()) {
            TTDataOffloadingNodeDispatcher::dispatch_enqueue_write_safe(
                language_extension_,
                stream,
                globals_stream,
                library_snippet_factory,
                src_name,
                dst_name,
                offloading_node.device_handle(),
                offloading_node.size(),
                offloading_node.blocking(),
                offloading_node.cq_no()
            );
        }
    }
}

codegen::InstrumentationInfo TTDataOffloadingNodeDispatcher::instrumentation_info() const {
    auto& tt_node = static_cast<const TTDataOffloadingNode&>(node_);
    if (tt_node.is_d2h()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_D2HTransfer,
            TargetType_Tenstorrent,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(tt_node.size())}}
        );
    } else if (tt_node.is_h2d()) {
        return codegen::InstrumentationInfo(
            node_.element_id(),
            codegen::ElementType_H2DTransfer,
            TargetType_Tenstorrent,
            analysis::LoopInfo{},
            {{"pcie_bytes", language_extension_.expression(tt_node.size())}}
        );
    } else {
        return codegen::LibraryNodeDispatcher::instrumentation_info();
    }
}

nlohmann::json TTDataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const TTDataOffloadingNode&>(library_node);
    nlohmann::json j;

    // Library node
    j["type"] = "library_node";
    j["element_id"] = library_node.element_id();

    // Debug info
    auto& debug_info = library_node.debug_info();
    j["has"] = debug_info.has();
    j["filename"] = debug_info.filename();
    j["start_line"] = debug_info.start_line();
    j["start_column"] = debug_info.start_column();
    j["end_line"] = debug_info.end_line();
    j["end_column"] = debug_info.end_column();

    // Library node properties
    j["code"] = std::string(library_node.code().value());

    // Offloading node properties
    j["blocking"] = node.blocking();
    j["cq_no"] = node.cq_no();
    j["device_handle"] = node.device_handle();
    sdfg::serializer::JSONSerializer serializer;
    j["size"] = serializer.expression(node.size());
    if (node.page_size().is_null()) {
        j["page_size"] = nlohmann::json::value_t::null;
    } else {
        j["page_size"] = serializer.expression(node.page_size());
    }
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

data_flow::LibraryNode& TTDataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Tenstorrent_Offloading.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    bool blocking = j.at("blocking").get<bool>();
    int cq_no = j.at("cq_no").get<int>();
    std::string device_handle = j.at("device_handle").get<std::string>();
    SymEngine::Expression size(j.at("size"));
    symbolic::Expression page_size;
    if (!j.contains("page_size") || j.at("page_size").is_null()) {
        page_size = SymEngine::null;
    } else {
        page_size = symbolic::parse(j.at("page_size"));
    }
    auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
    auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());

    return builder.add_library_node<TTDataOffloadingNode>(
        parent, debug_info, blocking, device_handle, size, page_size, transfer_direction, buffer_lifecycle, cq_no
    );
}

} // namespace tenstorrent
} // namespace sdfg
