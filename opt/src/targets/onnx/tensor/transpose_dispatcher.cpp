#include "sdfg/targets/onnx/tensor/transpose_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace tensor {

TransposeNodeDispatcher_ONNX::TransposeNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::TransposeNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), transpose_node_(node) {}

void TransposeNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string onnx_op = "Transpose";
    std::string node_name = "node_" + std::to_string(transpose_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_x = transpose_node_.input(0);
    const std::string& output_y = transpose_node_.output(0);

    // Get data type from the node
    auto prim_type = transpose_node_.primitive_type(data_flow_graph_);
    std::string onnx_type = primitive_type_to_onnx_type(prim_type);
    int onnx_elem_type = primitive_type_to_onnx_type_int(prim_type);

    // Emit ONNX runtime headers to globals (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // Build perm attribute
    const auto& perm = transpose_node_.perm();
    std::stringstream perm_ss;
    perm_ss << "[";
    for (size_t i = 0; i < perm.size(); ++i) {
        if (i > 0) perm_ss << ", ";
        perm_ss << perm[i];
    }
    perm_ss << "]";

    // Emit ONNX node definition to the graph snippet
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_x << "\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {" << std::endl;
    onnx_stream << "    \"perm\": " << perm_ss.str() << std::endl;
    onnx_stream << "  }" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX " << onnx_op << " operation" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Emit input shape array
    const auto& input_shape = transpose_node_.shape();
    stream << emit_shape_array(language_extension_, input_shape, node_name + "_input_shape") << std::endl;
    stream << "size_t " << node_name << "_input_ndim = " << input_shape.size() << ";" << std::endl;
    stream << "size_t " << node_name << "_input_size = " << emit_tensor_size(language_extension_, input_shape) << ";"
           << std::endl;
    stream << std::endl;

    // Calculate output shape based on permutation
    std::vector<symbolic::Expression> output_shape;
    for (int64_t p : perm) {
        if (p >= 0 && static_cast<size_t>(p) < input_shape.size()) {
            output_shape.push_back(input_shape[static_cast<size_t>(p)]);
        }
    }

    // Emit output shape array
    stream << emit_shape_array(language_extension_, output_shape, node_name + "_output_shape") << std::endl;
    stream << "size_t " << node_name << "_output_ndim = " << output_shape.size() << ";" << std::endl;
    stream << std::endl;

    // Emit perm array
    stream << "int64_t " << node_name << "_perm[] = {";
    for (size_t i = 0; i < perm.size(); ++i) {
        if (i > 0) stream << ", ";
        stream << perm[i];
    }
    stream << "};" << std::endl;
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

    // Create output tensor
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << output_y << ", " << node_name << "_input_size * sizeof(*" << output_y
           << ")," << std::endl;
    stream << "    " << node_name << "_output_shape, " << node_name << "_output_ndim, " << onnx_type << ", &"
           << node_name << "_output));" << std::endl;
    stream << std::endl;

    // Execute ONNX operation
    stream << "// Execute " << onnx_op << " via ONNX Runtime" << std::endl;
    stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\"};" << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_y << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input};" << std::endl;
    stream << "OrtValue* " << node_name << "_outputs[] = {" << node_name << "_output};" << std::endl;
    stream << std::endl;

    // Actually run the ONNX session for this model
    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << node_name
           << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 1, " << node_name
           << "_output_names, 1, " << node_name << "_outputs));" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace onnx
} // namespace sdfg
