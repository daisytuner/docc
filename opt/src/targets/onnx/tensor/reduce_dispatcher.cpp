#include "sdfg/targets/onnx/tensor/reduce_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace tensor {

ReduceNodeDispatcher_ONNX::ReduceNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::tensor::ReduceNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), reduce_node_(node) {}

void ReduceNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Get ONNX operation type from node code
    std::string onnx_op = get_onnx_op_type(reduce_node_.code().value());
    std::string node_name = "node_" + std::to_string(reduce_node_.element_id());

    // Get actual connector names from the node
    const std::string& input_x = reduce_node_.input(0);
    const std::string& output_y = reduce_node_.output(0);

    // Get data type from the node
    auto prim_type = reduce_node_.primitive_type(data_flow_graph_);
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

    // Build axes attribute string
    std::stringstream axes_ss;
    axes_ss << "[";
    const auto& axes = reduce_node_.axes();
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i > 0) axes_ss << ", ";
        axes_ss << axes[i];
    }
    axes_ss << "]";

    // Emit ONNX node definition to the graph snippet
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"" << onnx_op << "\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_x << "\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_y << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {" << std::endl;
    onnx_stream << "    \"axes\": " << axes_ss.str() << "," << std::endl;
    onnx_stream << "    \"keepdims\": " << (reduce_node_.keepdims() ? "1" : "0") << std::endl;
    onnx_stream << "  }" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX " << onnx_op << " operation" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Emit input shape array
    stream << emit_shape_array(language_extension_, reduce_node_.shape(), node_name + "_input_shape") << std::endl;
    stream << "size_t " << node_name << "_input_ndim = " << reduce_node_.shape().size() << ";" << std::endl;
    stream << "size_t " << node_name << "_input_size = " << emit_tensor_size(language_extension_, reduce_node_.shape())
           << ";" << std::endl;
    stream << std::endl;

    // Calculate output shape (removing or keeping reduced dimensions)
    const auto& input_shape = reduce_node_.shape();
    std::vector<symbolic::Expression> output_shape;

    // Normalize negative axes and determine output shape
    std::set<int64_t> reduce_axes;
    for (int64_t axis : axes) {
        if (axis < 0) {
            axis += static_cast<int64_t>(input_shape.size());
        }
        reduce_axes.insert(axis);
    }

    for (size_t i = 0; i < input_shape.size(); ++i) {
        bool is_reduced = reduce_axes.count(static_cast<int64_t>(i)) > 0;
        if (is_reduced) {
            if (reduce_node_.keepdims()) {
                output_shape.push_back(symbolic::integer(1));
            }
            // else skip this dimension
        } else {
            output_shape.push_back(input_shape[i]);
        }
    }

    if (output_shape.empty()) {
        output_shape.push_back(symbolic::integer(1)); // Scalar output
    }

    // Emit output shape array
    stream << emit_shape_array(language_extension_, output_shape, node_name + "_output_shape") << std::endl;
    stream << "size_t " << node_name << "_output_ndim = " << output_shape.size() << ";" << std::endl;
    stream << "size_t " << node_name << "_output_size = " << emit_tensor_size(language_extension_, output_shape) << ";"
           << std::endl;
    stream << std::endl;

    // Emit axes array for ONNX
    stream << "int64_t " << node_name << "_axes[] = {";
    for (size_t i = 0; i < axes.size(); ++i) {
        if (i > 0) stream << ", ";
        stream << axes[i];
    }
    stream << "};" << std::endl;
    stream << "size_t " << node_name << "_naxes = " << axes.size() << ";" << std::endl;
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
    stream << "    g_onnx_memory_info, " << output_y << ", " << node_name << "_output_size * sizeof(*" << output_y
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
