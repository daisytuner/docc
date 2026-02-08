#include "sdfg/targets/onnx/tensor/elementwise_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace tensor {

// ============================================================================
// ElementWiseUnaryNodeDispatcher_ONNX Implementation
// ============================================================================

ElementWiseUnaryNodeDispatcher_ONNX::ElementWiseUnaryNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::ElementWiseUnaryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), unary_node_(node) {}

void ElementWiseUnaryNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Get ONNX operation type from node code
    std::string onnx_op = get_onnx_op_type(unary_node_.code().value());
    std::string node_name = "node_" + std::to_string(unary_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_x = unary_node_.input(0);
    const std::string& output_y = unary_node_.output(0);

    // Get data type from the node
    auto prim_type = unary_node_.primitive_type(data_flow_graph_);
    std::string onnx_type = primitive_type_to_onnx_type(prim_type);
    int onnx_type_int = primitive_type_to_onnx_type_int(prim_type);

    // Emit ONNX runtime headers and init function to globals (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet (JSON format for Python post-processing)
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // Emit complete node definition including input/output tensor info
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_x << "\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_type_int << "," << std::endl;
    onnx_stream << "  \"attributes\": {}" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call that uses the pre-loaded session
    stream << "// ONNX " << onnx_op << " operation via ONNX Runtime session" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Emit shape array
    stream << emit_shape_array(language_extension_, unary_node_.shape(), node_name + "_shape") << std::endl;
    stream << "size_t " << node_name << "_ndim = " << unary_node_.shape().size() << ";" << std::endl;
    stream << "size_t " << node_name << "_size = " << emit_tensor_size(language_extension_, unary_node_.shape()) << ";"
           << std::endl;
    stream << std::endl;

    // Create input tensor
    stream << "OrtValue* " << node_name << "_input = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_x << ", " << node_name << "_size * sizeof(*" << input_x << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, " << node_name << "_ndim, " << onnx_type << ", &" << node_name
           << "_input));" << std::endl;
    stream << std::endl;

    // Create output tensor
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << output_y << ", " << node_name << "_size * sizeof(*" << output_y << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, " << node_name << "_ndim, " << onnx_type << ", &" << node_name
           << "_output));" << std::endl;
    stream << std::endl;

    // Run the ONNX session
    stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\"};" << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_y << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input};" << std::endl;
    stream << "OrtValue* " << node_name << "_outputs[] = {" << node_name << "_output};" << std::endl;
    stream << std::endl;

    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << std::endl;
    stream << "    " << node_name << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 1,"
           << std::endl;
    stream << "    " << node_name << "_output_names, 1, " << node_name << "_outputs));" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

// ============================================================================
// ElementWiseBinaryNodeDispatcher_ONNX Implementation
// ============================================================================

ElementWiseBinaryNodeDispatcher_ONNX::ElementWiseBinaryNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::ElementWiseBinaryNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), binary_node_(node) {}

void ElementWiseBinaryNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Get ONNX operation type from node code
    std::string onnx_op = get_onnx_op_type(binary_node_.code().value());
    std::string node_name = "node_" + std::to_string(binary_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_a = binary_node_.input(0);
    const std::string& input_b = binary_node_.input(1);
    const std::string& output_y = binary_node_.output(0);

    // Get data type from the node
    auto prim_type = binary_node_.primitive_type(data_flow_graph_);
    std::string onnx_type = primitive_type_to_onnx_type(prim_type);
    int onnx_type_int = primitive_type_to_onnx_type_int(prim_type);

    // Emit ONNX runtime headers and init function to globals (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet (JSON format for Python post-processing)
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // Emit complete node definition including input/output tensor info
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_a << "\", \"" << input_b << "\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_type_int << "," << std::endl;
    onnx_stream << "  \"attributes\": {}" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call that uses the pre-loaded session
    stream << "// ONNX " << onnx_op << " operation via ONNX Runtime session" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Emit shape array
    stream << emit_shape_array(language_extension_, binary_node_.shape(), node_name + "_shape") << std::endl;
    stream << "size_t " << node_name << "_ndim = " << binary_node_.shape().size() << ";" << std::endl;
    stream << "size_t " << node_name << "_size = " << emit_tensor_size(language_extension_, binary_node_.shape()) << ";"
           << std::endl;
    stream << std::endl;

    // Create input tensors
    stream << "OrtValue* " << node_name << "_input_a = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_a << ", " << node_name << "_size * sizeof(*" << input_a << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, " << node_name << "_ndim, " << onnx_type << ", &" << node_name
           << "_input_a));" << std::endl;
    stream << std::endl;

    stream << "OrtValue* " << node_name << "_input_b = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_b << ", " << node_name << "_size * sizeof(*" << input_b << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, " << node_name << "_ndim, " << onnx_type << ", &" << node_name
           << "_input_b));" << std::endl;
    stream << std::endl;

    // Run the ONNX session - let ONNX Runtime allocate output
    stream << "const char* " << node_name << "_input_names[] = {\"" << input_a << "\", \"" << input_b << "\"};"
           << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_y << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input_a, " << node_name << "_input_b};"
           << std::endl;
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << std::endl;

    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << std::endl;
    stream << "    " << node_name << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 2,"
           << std::endl;
    stream << "    " << node_name << "_output_names, 1, &" << node_name << "_output));" << std::endl;
    stream << std::endl;

    // Copy result from ONNX output tensor to our buffer
    stream << "{" << std::endl;
    stream << "    void* " << node_name << "_output_data = NULL;" << std::endl;
    stream << "    ORT_CHECK_STATUS(g_ort->GetTensorMutableData(" << node_name << "_output, &" << node_name
           << "_output_data));" << std::endl;
    stream << "    memcpy(" << output_y << ", " << node_name << "_output_data, " << node_name << "_size * sizeof(*"
           << output_y << "));" << std::endl;
    stream << "}" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input_a);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_input_b);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace onnx
} // namespace sdfg
