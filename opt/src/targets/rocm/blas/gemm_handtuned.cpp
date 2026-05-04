#include "sdfg/targets/rocm/blas/gemm_handtuned.h"

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg::rocm::blas {

GEMMNodeDispatcher_ROCMHandTuned::GEMMNodeDispatcher_ROCMHandTuned(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const math::blas::GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_ROCMHandTuned::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& gemm_node = static_cast<const math::blas::GEMMNode&>(this->node_);

    if (gemm_node.precision() != math::blas::BLAS_Precision::s) {
        throw std::runtime_error("Hand-tuned GEMM only supports single precision");
    }

    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");

    // Row-major to column-major swap (same convention as hipBLAS).
    // SDFG RowMajor: C(M,N) = alpha * op_a(A)(M,K) * op_b(B)(K,N) + beta * C
    // Column-major:  C(cm_M,cm_N) = alpha * op(first)(cm_M,K) * op(second)(K,cm_N) + beta * C
    // where cm_M = sdfg_N, cm_N = sdfg_M, first = B, second = A
    bool trans_first = (gemm_node.trans_b() != math::blas::BLAS_Transpose::No);
    bool trans_second = (gemm_node.trans_a() != math::blas::BLAS_Transpose::No);

    // Column-major dimensions
    std::string cm_M_str = this->language_extension_.expression(gemm_node.n());
    std::string cm_N_str = this->language_extension_.expression(gemm_node.m());
    std::string K_str = this->language_extension_.expression(gemm_node.k());

    // Column-major leading dimensions
    // first = B: no-trans → cm_M × K col-major (ld=cm_M=N); trans → K × cm_M col-major (ld=K)
    // second = A: no-trans → K × cm_N col-major (ld=K); trans → cm_N × K col-major (ld=cm_N=M)
    auto ld_first_sym = trans_first ? gemm_node.k() : gemm_node.n();
    auto ld_second_sym = trans_second ? gemm_node.m() : gemm_node.k();
    auto ldc_sym = gemm_node.n(); // C is cm_M × cm_N col-major, ld = cm_M = sdfg_N
    std::string ld_first_str = this->language_extension_.expression(ld_first_sym);
    std::string ld_second_str = this->language_extension_.expression(ld_second_sym);
    std::string ldc_str = this->language_extension_.expression(ldc_sym);

    // Tile parameters
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 16;
    const int BLOCK_X = 16; // tx walks cm_M (stride-1 in col-major C) for coalesced writes
    const int BLOCK_Y = 16; // ty walks cm_N
    // Strided sub-tile: thread (tx,ty) computes elements at
    //   rows: tx + i*BLOCK_X (i=0..TM-1) in cm_M
    //   cols: ty + j*BLOCK_Y (j=0..TN-1) in cm_N
    const int THREAD_TILE_M = TILE_M / BLOCK_X; // 4
    const int THREAD_TILE_N = TILE_N / BLOCK_Y; // 4

    std::string kernel_name = "sgemm_handtuned_" + std::to_string(gemm_node.element_id());

    // Forward declaration
    globals_stream << "__global__ void " << kernel_name << "(int cm_M, int cm_N, int K, float alpha, "
                   << "const float* __restrict__ first, int ld_first, "
                   << "const float* __restrict__ second, int ld_second, "
                   << "float beta, float* __restrict__ C, int ldc);" << std::endl;

    // Kernel body in separate compilation unit
    auto& ks = library_snippet_factory.require(kernel_name, "rocm.cpp", true).stream();
    ks << "#include " << library_snippet_factory.header_path().filename() << std::endl;
    ks << "#include <hip/hip_runtime.h>" << std::endl << std::endl;

    ks << "#define TILE_M " << TILE_M << std::endl;
    ks << "#define TILE_N " << TILE_N << std::endl;
    ks << "#define TILE_K " << TILE_K << std::endl;
    ks << "#define BLOCK_X " << BLOCK_X << std::endl;
    ks << "#define BLOCK_Y " << BLOCK_Y << std::endl;
    ks << "#define THREAD_TILE_M " << THREAD_TILE_M << std::endl;
    ks << "#define THREAD_TILE_N " << THREAD_TILE_N << std::endl;
    ks << std::endl;

    ks << "__global__ void " << kernel_name << "(int cm_M, int cm_N, int K, float alpha, "
       << "const float* __restrict__ first, int ld_first, "
       << "const float* __restrict__ second, int ld_second, "
       << "float beta, float* __restrict__ C, int ldc) {" << std::endl;
    ks.setIndent(4);

    // LDS: [TILE_K][TILE_M/N] layout for bank-conflict-free strided reads.
    // Threads in a warp read smFirst[kk][tx + i*BLOCK_X] with distinct tx → distinct banks.
    ks << "__shared__ float smFirst[TILE_K][TILE_M];" << std::endl;
    ks << "__shared__ float smSecond[TILE_K][TILE_N];" << std::endl;
    ks << std::endl;

    ks << "const int tx = threadIdx.x;" << std::endl;
    ks << "const int ty = threadIdx.y;" << std::endl;
    ks << "const int bx = blockIdx.x;" << std::endl;
    ks << "const int by = blockIdx.y;" << std::endl;
    ks << std::endl;

    // bx tiles cm_M, by tiles cm_N
    ks << "const int block_m = bx * TILE_M;" << std::endl;
    ks << "const int block_n = by * TILE_N;" << std::endl;
    ks << std::endl;

    // Accumulator registers (strided sub-tile)
    ks << "float acc[THREAD_TILE_M][THREAD_TILE_N];" << std::endl;
    ks << "for (int i = 0; i < THREAD_TILE_M; i++)" << std::endl;
    ks << "    for (int j = 0; j < THREAD_TILE_N; j++)" << std::endl;
    ks << "        acc[i][j] = 0.0f;" << std::endl;
    ks << std::endl;

    ks << "const int tid = ty * BLOCK_X + tx;" << std::endl;
    ks << std::endl;

    // K-loop
    ks << "for (int kt = 0; kt < K; kt += TILE_K) {" << std::endl;
    ks.setIndent(8);

    // --- Load first tile (cm_M × TILE_K) into smFirst[TILE_K][TILE_M] ---
    // 256 threads loading TILE_M * TILE_K = 1024 elements (4 per thread)
    ks << "// Load first tile (B) into shared memory" << std::endl;
    ks << "for (int load = tid; load < TILE_M * TILE_K; load += BLOCK_X * BLOCK_Y) {" << std::endl;
    ks.setIndent(12);
    if (trans_first) {
        // first is K × cm_M col-major → element (m,k) at first[k + m * ld_first]
        // K is stride-1 → iterate k-fast for coalesced global loads
        ks << "int lk = load % TILE_K;" << std::endl;
        ks << "int lm = load / TILE_K;" << std::endl;
    } else {
        // first is cm_M × K col-major → element (m,k) at first[m + k * ld_first]
        // cm_M is stride-1 → iterate m-fast for coalesced global loads
        ks << "int lm = load % TILE_M;" << std::endl;
        ks << "int lk = load / TILE_M;" << std::endl;
    }
    ks << "int gm = block_m + lm;" << std::endl;
    ks << "int gk = kt + lk;" << std::endl;
    if (trans_first) {
        ks << "smFirst[lk][lm] = (gm < cm_M && gk < K) ? first[gk + gm * ld_first] : 0.0f;" << std::endl;
    } else {
        ks << "smFirst[lk][lm] = (gm < cm_M && gk < K) ? first[gm + gk * ld_first] : 0.0f;" << std::endl;
    }
    ks.setIndent(8);
    ks << "}" << std::endl;
    ks << std::endl;

    // --- Load second tile (TILE_K × cm_N) into smSecond[TILE_K][TILE_N] ---
    ks << "// Load second tile (A) into shared memory" << std::endl;
    ks << "for (int load = tid; load < TILE_N * TILE_K; load += BLOCK_X * BLOCK_Y) {" << std::endl;
    ks.setIndent(12);
    if (trans_second) {
        // second is cm_N × K col-major → element (k,n) at second[n + k * ld_second]
        // cm_N is stride-1 → iterate n-fast for coalesced global loads
        ks << "int ln = load % TILE_N;" << std::endl;
        ks << "int lk = load / TILE_N;" << std::endl;
    } else {
        // second is K × cm_N col-major → element (k,n) at second[k + n * ld_second]
        // K is stride-1 → iterate k-fast for coalesced global loads
        ks << "int lk = load % TILE_K;" << std::endl;
        ks << "int ln = load / TILE_K;" << std::endl;
    }
    ks << "int gn = block_n + ln;" << std::endl;
    ks << "int gk = kt + lk;" << std::endl;
    if (trans_second) {
        ks << "smSecond[lk][ln] = (gn < cm_N && gk < K) ? second[gn + gk * ld_second] : 0.0f;" << std::endl;
    } else {
        ks << "smSecond[lk][ln] = (gn < cm_N && gk < K) ? second[gk + gn * ld_second] : 0.0f;" << std::endl;
    }
    ks.setIndent(8);
    ks << "}" << std::endl;
    ks << std::endl;

    ks << "__syncthreads();" << std::endl;
    ks << std::endl;

    // FMA accumulation with strided sub-tile access (bank-conflict-free)
    ks << "for (int kk = 0; kk < TILE_K; kk++) {" << std::endl;
    ks.setIndent(12);
    ks << "float a_frag[THREAD_TILE_M];" << std::endl;
    ks << "float b_frag[THREAD_TILE_N];" << std::endl;
    ks << "for (int i = 0; i < THREAD_TILE_M; i++)" << std::endl;
    ks << "    a_frag[i] = smFirst[kk][tx + i * BLOCK_X];" << std::endl;
    ks << "for (int j = 0; j < THREAD_TILE_N; j++)" << std::endl;
    ks << "    b_frag[j] = smSecond[kk][ty + j * BLOCK_Y];" << std::endl;
    ks << "for (int i = 0; i < THREAD_TILE_M; i++)" << std::endl;
    ks << "    for (int j = 0; j < THREAD_TILE_N; j++)" << std::endl;
    ks << "        acc[i][j] = __fmaf_rn(a_frag[i], b_frag[j], acc[i][j]);" << std::endl;
    ks.setIndent(8);
    ks << "}" << std::endl;
    ks << std::endl;
    ks << "__syncthreads();" << std::endl;

    // End K-loop
    ks.setIndent(4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Write results to C in column-major: C[row + col * ldc]
    // Adjacent tx → adjacent rows → coalesced writes in column-major
    ks << "for (int i = 0; i < THREAD_TILE_M; i++) {" << std::endl;
    ks.setIndent(8);
    ks << "int row = block_m + tx + i * BLOCK_X;" << std::endl;
    ks << "for (int j = 0; j < THREAD_TILE_N; j++) {" << std::endl;
    ks.setIndent(12);
    ks << "int col = block_n + ty + j * BLOCK_Y;" << std::endl;
    ks << "if (row < cm_M && col < cm_N) {" << std::endl;
    ks.setIndent(16);
    ks << "float val = alpha * acc[i][j];" << std::endl;
    ks << "if (beta != 0.0f) val += beta * C[row + col * ldc];" << std::endl;
    ks << "C[row + col * ldc] = val;" << std::endl;
    ks.setIndent(12);
    ks << "}" << std::endl;
    ks.setIndent(8);
    ks << "}" << std::endl;
    ks.setIndent(4);
    ks << "}" << std::endl;

    // End kernel
    ks.setIndent(0);
    ks << "}" << std::endl;

    // Launch code: guard + kernel invocation
    // first = __B, second = __A (row→col swap)
    stream << "if (" << cm_M_str << " != 0 && " << cm_N_str << " != 0 && " << K_str << " != 0) {" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "int _cm_M = (int)(" << cm_M_str << ");" << std::endl;
    stream << "int _cm_N = (int)(" << cm_N_str << ");" << std::endl;
    stream << "int _K = (int)(" << K_str << ");" << std::endl;
    // grid.x tiles cm_M (coalesced dim), grid.y tiles cm_N
    stream << "dim3 grid((_cm_M + " << (TILE_M - 1) << ") / " << TILE_M << ", (_cm_N + " << (TILE_N - 1) << ") / "
           << TILE_N << ", 1);" << std::endl;
    stream << "dim3 block(" << BLOCK_X << ", " << BLOCK_Y << ", 1);" << std::endl;
    stream << "hipLaunchKernelGGL(" << kernel_name << ", grid, block, 0, 0, "
           << "_cm_M, _cm_N, _K, __alpha, "
           << "__B, (int)(" << ld_first_str << "), "
           << "__A, (int)(" << ld_second_str << "), "
           << "__beta, __C, (int)(" << ldc_str << "));" << std::endl;

    check_rocm_kernel_launch_errors(stream, this->language_extension_);

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace sdfg::rocm::blas
