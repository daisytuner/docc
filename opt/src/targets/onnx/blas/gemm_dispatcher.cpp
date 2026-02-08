#include "sdfg/targets/onnx/blas/gemm_dispatcher.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/targets/onnx/onnx.h"

#include <stdexcept>

namespace sdfg {
namespace onnx {
namespace blas {

GEMMNodeDispatcher_ONNX::GEMMNodeDispatcher_ONNX(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), gemm_node_(node) {}

void GEMMNodeDispatcher_ONNX::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    std::string node_name = "node_" + std::to_string(gemm_node_.element_id());

    // Get ONNX type info from BLAS precision
    auto precision = gemm_node_.precision();
    std::string onnx_type = blas_precision_to_onnx_type(precision);
    int onnx_elem_type = blas_precision_to_onnx_type_int(precision);

    // Get transpose flags (ONNX uses 0/1 for transA/transB)
    int trans_a = (gemm_node_.trans_a() == math::blas::BLAS_Transpose::Trans ||
                   gemm_node_.trans_a() == math::blas::BLAS_Transpose::ConjTrans)
                      ? 1
                      : 0;
    int trans_b = (gemm_node_.trans_b() == math::blas::BLAS_Transpose::Trans ||
                   gemm_node_.trans_b() == math::blas::BLAS_Transpose::ConjTrans)
                      ? 1
                      : 0;

    // Find alpha and beta values from input edges
    // They must be ConstantNodes for ONNX target
    std::string alpha_value = "1.0";
    std::string beta_value = "0.0";

    for (auto& edge : data_flow_graph_.in_edges(gemm_node_)) {
        const std::string& conn = edge.dst_conn();
        if (conn == "__alpha" || conn == "__beta") {
            auto* src_node = dynamic_cast<const data_flow::ConstantNode*>(&edge.src());
            if (!src_node) {
                throw std::runtime_error(
                    "ONNX target requires " + conn +
                    " to be a compile-time constant. "
                    "Non-constant alpha/beta is not supported."
                );
            }
            if (conn == "__alpha") {
                alpha_value = src_node->data();
            } else {
                beta_value = src_node->data();
            }
        }
    }

    // Emit ONNX runtime base infrastructure (only once per SDFG)
    emit_onnx_runtime_init(stream, globals_stream, library_snippet_factory);

    // Model filename for this node
    std::string model_filename = "model_" + node_name + ".onnx";

    // Emit per-model session globals
    emit_onnx_model_session_globals(globals_stream, node_name, model_filename);

    // Get or create ONNX graph snippet
    auto& onnx_snippet = library_snippet_factory.require("model_" + node_name, "onnx.json", true);
    auto& onnx_stream = onnx_snippet.stream();

    // Emit ONNX Gemm node definition
    // ONNX Gemm: Y = alpha * A' * B' + beta * C
    // where A' = transpose(A) if transA else A
    onnx_stream << "{" << std::endl;
    onnx_stream << "  \"op_type\": \"Gemm\"," << std::endl;
    onnx_stream << "  \"name\": \"" << node_name << "\"," << std::endl;
    onnx_stream << "  \"inputs\": [\"__A\", \"__B\", \"__C\"]," << std::endl;
    onnx_stream << "  \"outputs\": [\"__C_out\"]," << std::endl;
    onnx_stream << "  \"elem_type\": " << onnx_elem_type << "," << std::endl;
    onnx_stream << "  \"attributes\": {" << std::endl;
    onnx_stream << "    \"alpha\": " << alpha_value << "," << std::endl;
    onnx_stream << "    \"beta\": " << beta_value << "," << std::endl;
    onnx_stream << "    \"transA\": " << trans_a << "," << std::endl;
    onnx_stream << "    \"transB\": " << trans_b << std::endl;
    onnx_stream << "  }" << std::endl;
    onnx_stream << "}," << std::endl;

    // Emit runtime call in the actual stream
    stream << "// ONNX Gemm operation" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    // Get dimensions
    std::string m_expr = language_extension_.expression(gemm_node_.m());
    std::string n_expr = language_extension_.expression(gemm_node_.n());
    std::string k_expr = language_extension_.expression(gemm_node_.k());

    // Emit shape arrays for A, B, C
    // A: [M, K] if no transpose, [K, M] if transpose
    // B: [K, N] if no transpose, [N, K] if transpose
    // C: [M, N]
    if (trans_a) {
        stream << "int64_t " << node_name << "_A_shape[] = {" << k_expr << ", " << m_expr << "};" << std::endl;
    } else {
        stream << "int64_t " << node_name << "_A_shape[] = {" << m_expr << ", " << k_expr << "};" << std::endl;
    }

    if (trans_b) {
        stream << "int64_t " << node_name << "_B_shape[] = {" << n_expr << ", " << k_expr << "};" << std::endl;
    } else {
        stream << "int64_t " << node_name << "_B_shape[] = {" << k_expr << ", " << n_expr << "};" << std::endl;
    }

    stream << "int64_t " << node_name << "_C_shape[] = {" << m_expr << ", " << n_expr << "};" << std::endl;
    stream << std::endl;

    // Calculate sizes
    if (trans_a) {
        stream << "size_t " << node_name << "_A_size = " << k_expr << " * " << m_expr << ";" << std::endl;
    } else {
        stream << "size_t " << node_name << "_A_size = " << m_expr << " * " << k_expr << ";" << std::endl;
    }

    if (trans_b) {
        stream << "size_t " << node_name << "_B_size = " << n_expr << " * " << k_expr << ";" << std::endl;
    } else {
        stream << "size_t " << node_name << "_B_size = " << k_expr << " * " << n_expr << ";" << std::endl;
    }

    stream << "size_t " << node_name << "_C_size = " << m_expr << " * " << n_expr << ";" << std::endl;
    stream << std::endl;

    // Initialize ONNX session for this model
    stream << "onnx_session_init_" << node_name << "();" << std::endl;
    stream << std::endl;

    // Create input tensor A
    stream << "OrtValue* " << node_name << "_input_A = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, __A, " << node_name << "_A_size * sizeof(*__A)," << std::endl;
    stream << "    " << node_name << "_A_shape, 2, " << onnx_type << ", &" << node_name << "_input_A));" << std::endl;
    stream << std::endl;

    // Create input tensor B
    stream << "OrtValue* " << node_name << "_input_B = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, __B, " << node_name << "_B_size * sizeof(*__B)," << std::endl;
    stream << "    " << node_name << "_B_shape, 2, " << onnx_type << ", &" << node_name << "_input_B));" << std::endl;
    stream << std::endl;

    // Create input tensor C (bias/accumulator)
    stream << "OrtValue* " << node_name << "_input_C = NULL;" << std::endl;
    stream << "ORT_CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(" << std::endl;
    stream << "    g_onnx_memory_info, __C, " << node_name << "_C_size * sizeof(*__C)," << std::endl;
    stream << "    " << node_name << "_C_shape, 2, " << onnx_type << ", &" << node_name << "_input_C));" << std::endl;
    stream << std::endl;

    // Run the ONNX session - let ONNX Runtime allocate output
    stream << "const char* " << node_name << "_input_names[] = {\"__A\", \"__B\", \"__C\"};" << std::endl;
    stream << "const char* " << node_name << "_output_names[] = {\"__C_out\"};" << std::endl;
    stream << "OrtValue* " << node_name << "_inputs[] = {" << node_name << "_input_A, " << node_name << "_input_B, "
           << node_name << "_input_C};" << std::endl;
    stream << "OrtValue* " << node_name << "_output = NULL;" << std::endl;
    stream << std::endl;

    stream << "ORT_CHECK_STATUS(g_ort->Run(g_onnx_session_" << node_name << ", NULL, " << std::endl;
    stream << "    " << node_name << "_input_names, (const OrtValue* const*)" << node_name << "_inputs, 3,"
           << std::endl;
    stream << "    " << node_name << "_output_names, 1, &" << node_name << "_output));" << std::endl;
    stream << std::endl;

    // Copy result from ONNX output tensor to C buffer
    stream << "{" << std::endl;
    stream << "    void* " << node_name << "_output_data = NULL;" << std::endl;
    stream << "    ORT_CHECK_STATUS(g_ort->GetTensorMutableData(" << node_name << "_output, &" << node_name
           << "_output_data));" << std::endl;
    stream << "    memcpy(__C, " << node_name << "_output_data, " << node_name << "_C_size * sizeof(*__C));"
           << std::endl;
    stream << "}" << std::endl;
    stream << std::endl;

    // Cleanup
    stream << "g_ort->ReleaseValue(" << node_name << "_input_A);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_input_B);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_input_C);" << std::endl;
    stream << "g_ort->ReleaseValue(" << node_name << "_output);" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace onnx
} // namespace sdfg
