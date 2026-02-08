#pragma once

#include <string>
#include <vector>

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace onnx {
namespace tensor {

/**
 * @brief Map primitive types to ONNX tensor element types (C enum)
 */
inline std::string primitive_type_to_onnx_type(types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Float:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT";
        case types::PrimitiveType::Double:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE";
        case types::PrimitiveType::Int8:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8";
        case types::PrimitiveType::Int16:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16";
        case types::PrimitiveType::Int32:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32";
        case types::PrimitiveType::Int64:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64";
        case types::PrimitiveType::UInt8:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8";
        case types::PrimitiveType::UInt16:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16";
        case types::PrimitiveType::UInt32:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32";
        case types::PrimitiveType::UInt64:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64";
        case types::PrimitiveType::Bool:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL";
        default:
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED";
    }
}

/**
 * @brief Map primitive types to ONNX tensor element type numbers for JSON
 */
inline int primitive_type_to_onnx_type_int(types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Float:
            return 1; // TensorProto.FLOAT
        case types::PrimitiveType::Double:
            return 11; // TensorProto.DOUBLE
        case types::PrimitiveType::Int8:
            return 3; // TensorProto.INT8
        case types::PrimitiveType::Int16:
            return 5; // TensorProto.INT16
        case types::PrimitiveType::Int32:
            return 6; // TensorProto.INT32
        case types::PrimitiveType::Int64:
            return 7; // TensorProto.INT64
        case types::PrimitiveType::UInt8:
            return 2; // TensorProto.UINT8
        case types::PrimitiveType::UInt16:
            return 4; // TensorProto.UINT16
        case types::PrimitiveType::UInt32:
            return 12; // TensorProto.UINT32
        case types::PrimitiveType::UInt64:
            return 13; // TensorProto.UINT64
        case types::PrimitiveType::Bool:
            return 9; // TensorProto.BOOL
        default:
            return 0; // TensorProto.UNDEFINED
    }
}

/**
 * @brief Map ONNX operation names
 */
inline std::string get_onnx_op_type(const std::string& node_code) {
    // Map internal codes to ONNX operator names
    if (node_code == "ml::Add") return "Add";
    if (node_code == "ml::Sub") return "Sub";
    if (node_code == "ml::Mul") return "Mul";
    if (node_code == "ml::Div") return "Div";
    if (node_code == "ml::Pow") return "Pow";
    if (node_code == "ml::Maximum") return "Max";
    if (node_code == "ml::Minimum") return "Min";
    if (node_code == "ml::Abs") return "Abs";
    if (node_code == "ml::Sqrt") return "Sqrt";
    if (node_code == "ml::Exp") return "Exp";
    if (node_code == "ml::Log") return "Log";
    if (node_code == "ml::Neg") return "Neg";
    if (node_code == "ml::Tanh") return "Tanh";
    if (node_code == "ml::Sigmoid") return "Sigmoid";
    if (node_code == "ml::ReLU") return "Relu";
    if (node_code == "ml::LeakyReLU") return "LeakyRelu";
    if (node_code == "ml::Elu") return "Elu";
    if (node_code == "ml::HardSigmoid") return "HardSigmoid";
    if (node_code == "ml::Erf") return "Erf";
    if (node_code == "ml::Cast") return "Cast";
    if (node_code == "ml::Fill") return "ConstantOfShape";
    if (node_code == "ml::Sum") return "ReduceSum";
    if (node_code == "ml::Mean") return "ReduceMean";
    if (node_code == "ml::Max") return "ReduceMax";
    if (node_code == "ml::Min") return "ReduceMin";
    if (node_code == "ml::Std") return "ReduceL2";
    if (node_code == "ml::Softmax") return "Softmax";
    if (node_code == "ml::Conv") return "Conv";
    if (node_code == "ml::Transpose") return "Transpose";
    if (node_code == "ml::Broadcast") return "Expand";
    return "Unknown";
}

/**
 * @brief Name of the sentinel snippet used to track ONNX runtime base init emission
 */
inline const std::string ONNX_RUNTIME_INIT_SNIPPET = "onnx_runtime_globals";

/**
 * @brief Generate ONNX runtime base initialization code (emitted once per compilation)
 *
 * This emits the shared ORT infrastructure (API, env, memory info) that is used by all models.
 * Uses the CodeSnippetFactory to ensure globals are emitted only once.
 */
inline void emit_onnx_runtime_init(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Check if we've already emitted the ONNX runtime globals by looking for the sentinel snippet
    auto it = library_snippet_factory.find(ONNX_RUNTIME_INIT_SNIPPET);
    if (it != library_snippet_factory.snippets().end()) {
        // Already emitted, nothing to do
        return;
    }

    // Create sentinel snippet to mark that we've emitted the globals
    library_snippet_factory.require(ONNX_RUNTIME_INIT_SNIPPET, "h", false);

    globals_stream << "#include <onnxruntime_c_api.h>" << std::endl;
    globals_stream << "#include <stdio.h>" << std::endl;
    globals_stream << "#include <stdlib.h>" << std::endl;
    globals_stream << "#include <string.h>" << std::endl;
    globals_stream << "#include <libgen.h>" << std::endl;
    globals_stream << "#include <dlfcn.h>" << std::endl;
    globals_stream << std::endl;

    // Shared ORT infrastructure (one per SDFG)
    globals_stream << "static const OrtApi* g_ort = NULL;" << std::endl;
    globals_stream << "static OrtEnv* g_ort_env = NULL;" << std::endl;
    globals_stream << "static OrtMemoryInfo* g_onnx_memory_info = NULL;" << std::endl;
    globals_stream << "static char g_onnx_lib_dir[4096] = {0};" << std::endl;
    globals_stream << std::endl;

    globals_stream << "#define ORT_CHECK_STATUS(status) \\" << std::endl;
    globals_stream << "    do { \\" << std::endl;
    globals_stream << "        if (status != NULL) { \\" << std::endl;
    globals_stream << "            const char* msg = g_ort->GetErrorMessage(status); \\" << std::endl;
    globals_stream << "            fprintf(stderr, \"ONNX Runtime error: %s\\n\", msg); \\" << std::endl;
    globals_stream << "            g_ort->ReleaseStatus(status); \\" << std::endl;
    globals_stream << "            exit(1); \\" << std::endl;
    globals_stream << "        } \\" << std::endl;
    globals_stream << "    } while (0)" << std::endl;
    globals_stream << std::endl;

    // Emit helper function to find library directory
    globals_stream << "static void onnx_find_lib_dir() {" << std::endl;
    globals_stream << "    if (g_onnx_lib_dir[0] != '\\0') return;" << std::endl;
    globals_stream << "    Dl_info info;" << std::endl;
    globals_stream << "    if (dladdr((void*)onnx_find_lib_dir, &info) && info.dli_fname) {" << std::endl;
    globals_stream << "        char* path_copy = strdup(info.dli_fname);" << std::endl;
    globals_stream << "        char* dir = dirname(path_copy);" << std::endl;
    globals_stream << "        strncpy(g_onnx_lib_dir, dir, sizeof(g_onnx_lib_dir) - 1);" << std::endl;
    globals_stream << "        free(path_copy);" << std::endl;
    globals_stream << "    }" << std::endl;
    globals_stream << "}" << std::endl;
    globals_stream << std::endl;

    // Emit base initialization function (shared ORT infrastructure)
    globals_stream << "static void onnx_runtime_base_init() {" << std::endl;
    globals_stream << "    if (g_ort != NULL) return;" << std::endl;
    globals_stream << "    onnx_find_lib_dir();" << std::endl;
    globals_stream << "    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);" << std::endl;
    globals_stream << "    ORT_CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, \"docc_onnx\", &g_ort_env));"
                   << std::endl;
    globals_stream << "    ORT_CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, "
                      "&g_onnx_memory_info));"
                   << std::endl;
    globals_stream << "}" << std::endl;
    globals_stream << std::endl;
}

/**
 * @brief Generate per-model session initialization code
 *
 * Each ONNX model gets its own session, session options, and model path.
 * The session is lazily initialized on first use.
 *
 * @param globals_stream Stream for global declarations
 * @param node_name Unique name for this node/model
 * @param model_filename Name of the ONNX model file (e.g., "model_node_123.onnx")
 */
inline void emit_onnx_model_session_globals(
    codegen::PrettyPrinter& globals_stream, const std::string& node_name, const std::string& model_filename
) {
    // Per-model session globals
    globals_stream << "static OrtSession* g_onnx_session_" << node_name << " = NULL;" << std::endl;
    globals_stream << "static OrtSessionOptions* g_onnx_session_options_" << node_name << " = NULL;" << std::endl;
    globals_stream << "static char g_onnx_model_path_" << node_name << "[4096] = {0};" << std::endl;
    globals_stream << std::endl;

    // Per-model initialization function
    globals_stream << "static void onnx_session_init_" << node_name << "() {" << std::endl;
    globals_stream << "    if (g_onnx_session_" << node_name << " != NULL) return;" << std::endl;
    globals_stream << "    onnx_runtime_base_init();" << std::endl;
    globals_stream << "    snprintf(g_onnx_model_path_" << node_name << ", sizeof(g_onnx_model_path_" << node_name
                   << "), \"%s/" << model_filename << "\", g_onnx_lib_dir);" << std::endl;
    globals_stream << "    ORT_CHECK_STATUS(g_ort->CreateSessionOptions(&g_onnx_session_options_" << node_name << "));"
                   << std::endl;
    globals_stream << "    g_ort->SetIntraOpNumThreads(g_onnx_session_options_" << node_name << ", 1);" << std::endl;
    globals_stream << "    g_ort->SetSessionGraphOptimizationLevel(g_onnx_session_options_" << node_name
                   << ", ORT_ENABLE_ALL);" << std::endl;
    globals_stream << "    ORT_CHECK_STATUS(g_ort->CreateSession(g_ort_env, g_onnx_model_path_" << node_name
                   << ", g_onnx_session_options_" << node_name << ", &g_onnx_session_" << node_name << "));"
                   << std::endl;
    globals_stream << "}" << std::endl;
    globals_stream << std::endl;
}

/**
 * @brief Generate shape array declaration for ONNX tensors
 */
inline std::string emit_shape_array(
    codegen::LanguageExtension& language_extension,
    const std::vector<symbolic::Expression>& shape,
    const std::string& array_name
) {
    std::stringstream ss;
    ss << "int64_t " << array_name << "[] = {";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << language_extension.expression(shape[i]);
    }
    ss << "};";
    return ss.str();
}

/**
 * @brief Calculate total tensor size from shape
 */
inline std::string
emit_tensor_size(codegen::LanguageExtension& language_extension, const std::vector<symbolic::Expression>& shape) {
    if (shape.empty()) {
        return "1";
    }

    symbolic::Expression total = shape[0];
    for (size_t i = 1; i < shape.size(); ++i) {
        total = symbolic::mul(total, shape[i]);
    }
    return language_extension.expression(total);
}

} // namespace tensor
} // namespace onnx
} // namespace sdfg
