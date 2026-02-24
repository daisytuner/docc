#include "sdfg/data_flow/library_nodes/load_const_node.h"

#include "daisy_rtl/base64.h"

namespace sdfg {
namespace data_flow {

LoadConstNode::LoadConstNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    std::unique_ptr<types::IType> type,
    std::unique_ptr<ConstSource> data_source
)
    : LibraryNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LoadConst,
          {"_out"},
          {},
          true,
          data_flow::ImplementationType_NONE
      ),
      type_(std::move(type)), data_source_(std::move(data_source)) {}

symbolic::SymbolSet LoadConstNode::symbols() const { return {}; }

const types::IType& LoadConstNode::type() const { return *type_; }

ConstSource& LoadConstNode::data_source() const { return *data_source_; }

std::string LoadConstNode::toStr() const { return "LoadConst(" + type_->print() + ": " + data_source_->toStr() + ")"; }

std::unique_ptr<data_flow::DataFlowNode> LoadConstNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<
        LoadConstNode>(element_id, debug_info_, vertex, parent, type_->clone(), data_source_->clone());
}

void LoadConstNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {}

nlohmann::json LoadConstNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const LoadConstNode& node = static_cast<const LoadConstNode&>(library_node);

    nlohmann::json j;
    j["code"] = node.code().value();
    serializer::JSONSerializer serializer;
    serializer.type_to_json(j["load_type"], node.type());
    const_source_to_json(j["source"], node.data_source());

    return j;
}

void LoadConstNodeSerializer::const_source_to_json(nlohmann::json& j, const ConstSource& source) {
    if (auto* inMemSource = dynamic_cast<const InMemoryConstSource*>(&source)) {
        j["source_type"] = "in_memory";
        j["data"] = base64_encode(inMemSource->data_.data(), inMemSource->data_.size());
    } else if (auto* rawFileSource = dynamic_cast<const RawFileConstSource*>(&source)) {
        j["source_type"] = "raw_file";
        j["filename"] = rawFileSource->filename().string();
        j["size"] = rawFileSource->num_bytes();
    } else {
        throw InvalidSDFGException("Invalid const source type");
    }
}

std::unique_ptr<ConstSource> LoadConstNodeSerializer::json_to_const_source(const nlohmann::json& j) {
    std::string source_type = j.at("source_type").get<std::string>();

    if (source_type == "in_memory") {
        auto decoded = base64_decode(j.at("data").get<std::string>());
        auto ptr = std::make_unique<InMemoryConstSource>(decoded.size());
        ptr->data_ = std::move(decoded);
        return ptr;
    } else if (source_type == "raw_file") {
        auto filename = j.at("filename").get<std::filesystem::path>();
        auto size = j.at("size").get<size_t>();
        return std::make_unique<RawFileConstSource>(filename, size);
    } else {
        throw InvalidSDFGException("Invalid const source type '" + source_type + "'");
    }
}

data_flow::LibraryNode& LoadConstNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j.at("code").get<std::string>();
    if (code != LibraryNodeType_LoadConst.value()) {
        throw InvalidSDFGException("Invalid library node code");
    }

    std::cout << "Loading const node" << std::endl;

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));
    std::unique_ptr<types::IType> type = serializer.json_to_type(j.at("load_type"));

    std::unique_ptr<ConstSource> data_source = json_to_const_source(j.at("source"));

    return builder.add_library_node<LoadConstNode>(parent, debug_info, std::move(type), std::move(data_source));
}

LoadConstNodeDispatcher::LoadConstNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const LoadConstNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void LoadConstNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string local_ptr = node_.outputs()[0];
    auto& node = static_cast<const LoadConstNode&>(node_);
    auto& src = node.data_source();
    if (auto* inMemSrc = dynamic_cast<const InMemoryConstSource*>(&src)) {
        dispatch_inline_const(stream, globals_stream, library_snippet_factory, *inMemSrc, local_ptr);
    } else if (auto* rawFileSrc = dynamic_cast<const RawFileConstSource*>(&src)) {
        dispatch_runtime_load(stream, globals_stream, library_snippet_factory, *rawFileSrc, local_ptr);
    } else {
        throw InvalidSDFGException("Invalid const source type");
    }
}

void LoadConstNodeDispatcher::dispatch_inline_const(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const ConstSource& source,
    const std::string& output_container
) {
    std::string identifier = "daisy_load_const_" + std::to_string(node_.element_id());
    globals_stream << "static const uint8_t " << identifier << "[] = {";

    auto it = source.inline_begin();
    auto end = source.inline_end();
    bool first = true;
    while (it != end) {
        if (first) {
            first = false;
        } else {
            globals_stream << ", ";
        }
        globals_stream << "0x" << std::hex << +*it << std::dec;
        ++it;
    }
    globals_stream << "};" << std::endl;

    auto& node = dynamic_cast<const LoadConstNode&>(node_);

    auto target_type = language_extension_.declaration("", node.type());
    stream << output_container << " = " << "const_cast<" << target_type << ">(reinterpret_cast<const " << target_type
           << ">(&" + identifier + "[0]));" << std::endl;
}

void LoadConstNodeDispatcher::
    ensure_arg_capture_dep(codegen::PrettyPrinter& globals_stream, codegen::CodeSnippetFactory& code_snippet_factory) {
    auto marker = "arg_capture_includes"; // TODO use modern api to require linking of

    if (code_snippet_factory.find(marker) == code_snippet_factory.snippets().end()) {
        code_snippet_factory.require(marker, "", false);
        // hacky way to only emit to global once
        globals_stream << "#include <daisy_rtl/arg_capture_io.h>" << std::endl;

        globals_stream << std::endl;
    }
}

void LoadConstNodeDispatcher::dispatch_runtime_load(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const RawFileConstSource& source,
    const std::string& output_container
) {
    ensure_arg_capture_dep(globals_stream, library_snippet_factory);

    stream << "arg_capture::ArgCaptureIO::read_data_from_raw_file(" << source.filename().string() << ", "
           << output_container << ", " << source.num_bytes() << ");" << std::endl;
}

} // namespace data_flow
} // namespace sdfg
