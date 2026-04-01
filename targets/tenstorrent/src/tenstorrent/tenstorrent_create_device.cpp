#include "docc/target/tenstorrent/tenstorrent_create_device.h"

#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/structured_control_flow/block.h>

#include "sdfg/data_flow/library_node.h"

namespace sdfg {
namespace tenstorrent {

TTCreateDevice::TTCreateDevice(
    size_t element_id,
    const DebugInfo &debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph &parent,
    const std::vector<std::string> &outputs,
    symbolic::Expression device_id
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Tenstorrent_CreateDevice,
          outputs,
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      device_id_(device_id) {}

void TTCreateDevice::validate(const Function &function) const {
    // TODO: Implement
}

const symbolic::Expression TTCreateDevice::device_id() const { return this->device_id_; }

std::unique_ptr<data_flow::DataFlowNode> TTCreateDevice::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph &parent) const {
    return std::make_unique<TTCreateDevice>(element_id, debug_info_, vertex, parent, outputs_, this->device_id_);
}

void TTCreateDevice::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    device_id_ = symbolic::subs(device_id_, old_expression, new_expression);
}

symbolic::SymbolSet TTCreateDevice::symbols() const {
    auto atoms_id = symbolic::atoms(device_id_);
    symbolic::SymbolSet atoms;
    atoms.insert(atoms_id.begin(), atoms_id.end());
    return atoms;
}

TTCreateDeviceDispatcher::TTCreateDeviceDispatcher(
    codegen::LanguageExtension &language_extension,
    const Function &function,
    const data_flow::DataFlowGraph &data_flow_graph,
    const data_flow::LibraryNode &node
)
    : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void TTCreateDeviceDispatcher::dispatch(
    codegen::PrettyPrinter &stream,
    codegen::PrettyPrinter &globals_stream,
    codegen::CodeSnippetFactory &library_snippet_factory
) {
    // Assumes node_ is of type CUDAMalloc
    auto &tt_node = static_cast<const TTCreateDevice &>(node_);
    // Change the targeted device ID if necessary
    stream << "CreateDevice(" << language_extension_.expression(tt_node.device_id()) << ");" << std::endl;
}

// CUDAMallocSerializer Implementation

nlohmann::json TTCreateDeviceSerializer::serialize(const sdfg::data_flow::LibraryNode &library_node) {
    const auto &node = static_cast<const TTCreateDevice &>(library_node);
    nlohmann::json j;

    // Library node properties
    j["code"] = std::string(node.code().value());
    j["outputs"] = node.outputs();

    // CUDAMalloc specific properties
    sdfg::serializer::JSONSerializer serializer;
    j["device_id"] = serializer.expression(node.device_id());

    return j;
}

data_flow::LibraryNode &TTCreateDeviceSerializer::deserialize(
    const nlohmann::json &j, sdfg::builder::StructuredSDFGBuilder &builder, sdfg::structured_control_flow::Block &parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("outputs"));
    assert(j.contains("device_id"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_Tenstorrent_CreateDevice.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Extract properties
    auto outputs = j.at("outputs").get<std::vector<std::string> >();

    // Deserialize expressions using serializer
    SymEngine::Expression device_id(j.at("device_id"));

    return builder.add_library_node<TTCreateDevice>(parent, debug_info, outputs, device_id);
}
} // namespace tenstorrent
} // namespace sdfg
