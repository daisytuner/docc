#include "sdfg/targets/onnx/tensor/conv_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace tensor {

ConvNodeDispatcher_ONNX::ConvNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::ConvNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), conv_node_(node) {}

void ConvNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string onnx_op = "Conv";
    std::string node_name = "node_" + std::to_string(conv_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_x = conv_node_.input(0); // X
    const std::string& input_w = conv_node_.input(1); // W
    // B is optional - conv_node_ may have 2 or 3 inputs
    bool has_bias = conv_node_.inputs().size() > 2;
    std::string input_b = has_bias ? conv_node_.input(2) : "";
    const std::string& output_y = conv_node_.output(0); // Y

    // Get data type from the node
    auto prim_type = conv_node_.primitive_type(data_flow_graph_);
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

    // Build attributes for ONNX Conv node
    const auto& kernel_shape = conv_node_.kernel_shape();
    const auto& strides = conv_node_.strides();
    const auto& pads = conv_node_.pads();
    const auto& dilations = conv_node_.dilations();

    // Emit kernel_shape attribute
    std::stringstream kernel_ss;
    kernel_ss << "[";
    for (size_t i = 0; i < kernel_shape.size(); ++i) {
        if (i > 0) kernel_ss << ", ";
        kernel_ss << language_extension_.expression(kernel_shape[i]);
    }
    kernel_ss << "]";

    // Emit strides attribute
    std::stringstream strides_ss;
    strides_ss << "[";
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i > 0) strides_ss << ", ";
        strides_ss << language_extension_.expression(strides[i]);
    }
    strides_ss << "]";

    // Emit pads attribute (ONNX expects [begin_h, begin_w, end_h, end_w] for 2D)
    std::stringstream pads_ss;
    pads_ss << "[";
    for (size_t i = 0; i < pads.size(); ++i) {
        if (i > 0) pads_ss << ", ";
        pads_ss << language_extension_.expression(pads[i]);
    }
    pads_ss << "]";

    // Emit dilations attribute
    std::stringstream dilations_ss;
    dilations_ss << "[";
    for (size_t i = 0; i < dilations.size(); ++i) {
        if (i > 0) dilations_ss << ", ";
        dilations_ss << language_extension_.expression(dilations[i]);
    }
    dilations_ss << "]";

    // Emit ONNX node definition to the graph snippet
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    if (has_bias) {
        onnx_stream << "  \"inputs\": [\"" << input_x << "\", \"" << input_w << "\", \"" << input_b << "\"],"
                    << std::endl;
    } else {
        onnx_stream << "  \"inputs\": [\"" << input_x << "\", \"" << input_w << "\"]," << std::endl;
    }
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {" << std::endl;
    onnx_stream << "    \"kernel_shape\": " << kernel_ss.str() << "," << std::endl;
    onnx_stream << "    \"strides\": " << strides_ss.str() << "," << std::endl;
    onnx_stream << "    \"pads\": " << pads_ss.str() << "," << std::endl;
    onnx_stream << "    \"dilations\": " << dilations_ss.str() << "," << std::endl;
    onnx_stream << "    \"group\": " << language_extension_.expression(conv_node_.group()) << std::endl;
    onnx_stream << "  }" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX " << onnx_op << " operation" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Emit input shape array [N, C_in, D1, ..., Dn]
    stream << emit_shape_array(language_extension_, conv_node_.shape(), node_name + "_input_shape") << std::endl;
    stream << "size_t " << node_name << "_input_ndim = " << conv_node_.shape().size() << ";" << std::endl;
    stream << "size_t " << node_name << "_input_size = " << emit_tensor_size(language_extension_, conv_node_.shape())
           << ";" << std::endl;
    stream << std::endl;

    // Emit kernel shape array
    stream << emit_shape_array(language_extension_, kernel_shape, node_name + "_kernel_shape") << std::endl;
    stream << std::endl;

    // Emit strides array
    stream << "int64_t " << node_name << "_strides[] = {";
    for (size_t i = 0; i < strides.size(); ++i) {
        if (i > 0) stream << ", ";
        stream << language_extension_.expression(strides[i]);
    }
    stream << "};" << std::endl;

    // Emit pads array
    stream << "int64_t " << node_name << "_pads[] = {";
    for (size_t i = 0; i < pads.size(); ++i) {
        if (i > 0) stream << ", ";
        stream << language_extension_.expression(pads[i]);
    }
    stream << "};" << std::endl;

    // Emit dilations array
    stream << "int64_t " << node_name << "_dilations[] = {";
    for (size_t i = 0; i < dilations.size(); ++i) {
        if (i > 0) stream << ", ";
        stream << language_extension_.expression(dilations[i]);
    }
    stream << "};" << std::endl;
    stream << std::endl;

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Create input tensor X using global memory info
    stream << "OrtValue* " << node_name << "_input_x = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_x << ", " << node_name << "_input_size * sizeof(*" << input_x << "),"
           << std::endl;
    stream << "    " << node_name << "_input_shape, " << node_name << "_input_ndim, " << onnx_type << ", &" << node_name
           << "_input_x));" << std::endl;
    stream << std::endl;

    // Weight tensor W
    stream << "// Weight tensor " << input_w << ": [C_out, C_in/group, k1, k2, ...]" << std::endl;
    stream << "OrtValue* " << node_name << "_input_w = NULL;  // Initialize from " << input_w << " data" << std::endl;
    stream << std::endl;

    // Bias tensor (optional)
    if (has_bias) {
        stream << "// Bias tensor " << input_b << ": [C_out]" << std::endl;
        stream << "OrtValue* " << node_name << "_input_b = NULL;  // Initialize from " << input_b << " data"
               << std::endl;
    } else {
        stream << "// No bias tensor" << std::endl;
        stream << "OrtValue* " << node_name << "_input_b = NULL;" << std::endl;
    }
    stream << std::endl;

    // Create output tensor Y
    stream << "// Output tensor " << output_y << ": computed shape based on input, kernel, strides, pads" << std::endl;
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << std::endl;

    // Execute ONNX operation
    stream << "// Execute " << onnx_op << " via ONNX Runtime" << std::endl;
    if (has_bias) {
        stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\", \"" << input_w << "\", \""
               << input_b << "\"};" << std::endl;
        stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input_x, " << node_name << "_input_w, "
               << node_name << "_input_b};" << std::endl;
    } else {
        stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\", \"" << input_w << "\"};"
               << std::endl;
        stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input_x, " << node_name << "_input_w};"
               << std::endl;
    }
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_y << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_outputs[] = {" << node_name << "_output};" << std::endl;
    stream << std::endl;

    // Actually run the ONNX session for this model
    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << node_name
           << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, " << (has_bias ? "3" : "2") << ", "
           << node_name << "_output_names, 1, " << node_name << "_outputs));" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input_x);" << std::endl;
    stream << "if (" << node_name << "_input_w) g_ort->ReleaseValue(" << node_name << "_input_w);" << std::endl;
    stream << "if (" << node_name << "_input_b) g_ort->ReleaseValue(" << node_name << "_input_b);" << std::endl;
    stream << "if (" << node_name << "_output) g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace tensor
} // namespace onnx
} // namespace sdfg
