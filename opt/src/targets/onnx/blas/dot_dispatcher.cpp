#include "sdfg/targets/onnx/blas/dot_dispatcher.h"

#include "sdfg/targets/onnx/onnx.h"

namespace sdfg {
namespace onnx {
namespace blas {

DotNodeDispatcher_ONNX::DotNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::DotNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), dot_node_(node) {}

void DotNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string node_name = "node_" + std::to_string(dot_node_.element_id());

    // Get input/output connector names
    const std::string& input_x = dot_node_.inputs().at(0);
    const std::string& input_y = dot_node_.inputs().at(1);
    const std::string& output_result = dot_node_.outputs().at(0);

    // Get ONNX type info from BLAS precision
    auto precision = dot_node_.precision();
    std::string onnx_type = blas_precision_to_onnx_type(precision);
    int onnx_elem_type = blas_precision_to_onnx_type_int(precision);

    // Emit ONNX runtime base infrastructure (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet
    // DOT product in ONNX: Mul(x, y) -> ReduceSum -> result
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // We need two operations: Mul followed by ReduceSum
    // For simplicity, we'll emit them as separate nodes in the graph

    // Node 1: Element-wise multiply
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"Mul\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "_mul\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << input_x << "\", \"" << input_y << "\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << node_name << "_prod\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {}" << std::endl;
    onnx_stream << "}," << std::endl;

    // Node 2: ReduceSum over all dimensions
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"ReduceSum\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "_sum\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"" << node_name << "_prod\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"" << output_result << "\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {" << std::endl;
    onnx_stream << "    \"keepdims\": 0" << std::endl;
    onnx_stream << "  }" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX Dot product operation (Mul + ReduceSum)" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Get vector length
    std::string n_expr = language_extension_.expression(dot_node_.n());

    // Emit shape array for input vectors
    stream << "int64_t " << node_name << "_shape[] = {" << n_expr << "};" << std::endl;
    stream << "size_t " << node_name << "_size = " << n_expr << ";" << std::endl;
    stream << std::endl;

    // Output is a scalar
    stream << "int64_t " << node_name << "_output_shape[] = {1};" << std::endl;
    stream << std::endl;

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Create input tensor X
    stream << "OrtValue* " << node_name << "_input_x = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_x << ", " << node_name << "_size * sizeof(*" << input_x << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, 1, " << onnx_type << ", &" << node_name << "_input_x));" << std::endl;
    stream << std::endl;

    // Create input tensor Y
    stream << "OrtValue* " << node_name << "_input_y = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, " << input_y << ", " << node_name << "_size * sizeof(*" << input_y << "),"
           << std::endl;
    stream << "    " << node_name << "_shape, 1, " << onnx_type << ", &" << node_name << "_input_y));" << std::endl;
    stream << std::endl;

    // Run the ONNX session - let ONNX Runtime allocate output
    stream << "const char* " << node_name << "_input_names[] = {\"" << input_x << "\", \"" << input_y << "\"};"
           << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"" << output_result << "\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input_x, " << node_name << "_input_y};"
           << std::endl;
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << std::endl;

    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << std::endl;
    stream << "    " << node_name << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 2,"
           << std::endl;
    stream << "    " << node_name << "_output_names, 1, &" << node_name << "_output));" << std::endl;
    stream << std::endl;

    // Copy scalar result from ONNX output tensor
    stream << "{" << std::endl;
    stream << "    void* " << node_name << "_output_data = NULL;" << std::endl;
    stream << "    ORT_CHECK_STATUS(g_ort->GetTensorMutableData(" << node_name << "_output, &" << node_name
           << "_output_data));" << std::endl;
    stream << "    " << output_result << " = *((" << std::endl;
    // Determine the correct C type based on precision
    switch (precision) {
        case math::blas::BLAS_Precision::s:
            stream << "        float";
            break;
        case math::blas::BLAS_Precision::d:
            stream << "        double";
            break;
        case math::blas::BLAS_Precision::h:
            stream << "        _Float16";
            break;
        default:
            stream << "        float";
    }
    stream << "*)" << node_name << "_output_data);" << std::endl;
    stream << "}" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input_x);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_input_y);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace onnx
} // namespace sdfg
