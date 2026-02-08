#include "sdfg/targets/onnx/tensor/broadcast_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace tensor {

BroadcastNodeDispatcher_ONNX::BroadcastNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::BroadcastNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), broadcast_node_(node) {}

void BroadcastNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string onnx_op = "Expand"; // ONNX uses Expand for broadcasting
    std::string node_name = "node_" + std::to_string(broadcast_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_x = broadcast_node_.input(0);
    const std::string& output_y = broadcast_node_.output(0);

    // Get data type from the node
    auto prim_type = broadcast_node_.primitive_type(data_flow_graph_);
    std::string onnx_type = primitive_type_to_onnx_type(prim_type);
    int onnx_elem_type = primitive_type_to_onnx_type_int(prim_type);

    // Emit ONNX runtime base infrastructure (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // Emit ONNX node definition to the graph snippet
    // ONNX Expand takes input tensor and a shape tensor
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_x << "\", \"shape\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {}" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX " << onnx_op << " operation (broadcast)" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Emit input shape array
    const auto& input_shape = broadcast_node_.input_shape();
    stream << emit_shape_array(language_extension_, input_shape, node_name + "_input_shape") << std::endl;
    stream << "size_t " << node_name << "_input_ndim = " << input_shape.size() << ";" << std::endl;
    stream << "size_t " << node_name << "_input_size = " << emit_tensor_size(language_extension_, input_shape) << ";"
           << std::endl;
    stream << std::endl;

    // Emit output shape array
    const auto& output_shape = broadcast_node_.output_shape();
    stream << emit_shape_array(language_extension_, output_shape, node_name + "_output_shape") << std::endl;
    stream << "size_t " << node_name << "_output_ndim = " << output_shape.size() << ";" << std::endl;
    stream << "size_t " << node_name << "_output_size = " << emit_tensor_size(language_extension_, output_shape) << ";"
           << std::endl;
    stream << std::endl;

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Create input tensor using global memory info
    stream << "OrtValue* " << node_name << "_input = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_x << ", " << node_name << "_input_size * sizeof(*" << input_x << "),"
           << std::endl;
    stream << "    " << node_name << "_input_shape, " << node_name << "_input_ndim, " << onnx_type << ", &" << node_name
           << "_input));" << std::endl;
    stream << std::endl;

    // Create shape tensor for ONNX Expand (the target shape as int64 tensor)
    stream << "// Shape tensor for Expand operation" << std::endl;
    stream << "OrtValue* " << node_name << "_shape_tensor = NULL;" << std::endl;
    stream << "int64_t " << node_name << "_shape_dims[] = {" << output_shape.size() << "};" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << node_name << "_output_shape, " << output_shape.size()
           << " * sizeof(int64_t)," << std::endl;
    stream << "    " << node_name << "_shape_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &" << node_name
           << "_shape_tensor));" << std::endl;
    stream << std::endl;

    // Create output tensor
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << output_y << ", " << node_name << "_output_size * sizeof(*" << output_y
           << ")," << std::endl;
    stream << "    " << node_name << "_output_shape, " << node_name << "_output_ndim, " << onnx_type << ", &"
           << node_name << "_output));" << std::endl;
    stream << std::endl;

    // Execute ONNX operation
    stream << "// Execute " << onnx_op << " via ONNX Runtime" << std::endl;
    stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\", \"shape\"};" << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_y << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input, " << node_name << "_shape_tensor};"
           << std::endl;
    stream << "OrtValue* " << node_name << "_outputs[] = {" << node_name << "_output};" << std::endl;
    stream << std::endl;

    // Actually run the ONNX session for this model
    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << node_name
           << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 2, " << node_name
           << "_output_names, 1, " << node_name << "_outputs));" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_shape_tensor);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace onnx
} // namespace sdfg
