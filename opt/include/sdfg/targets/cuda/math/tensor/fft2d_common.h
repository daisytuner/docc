/**
 * @file fft2d_common.h
 * @brief Shared hand-tuned mixed-radix Stockham FFT kernel emission for the
 *        @ref R2CFFT2DNode and @ref C2RFFT2DNode CUDA dispatchers.
 *
 * Emits the same validated R2C/C2R kernels as the fused fft_conv dispatcher
 * (mixed-radix Stockham, half-width spectrum, conditional shared-memory padding),
 * minus the pad/crop/complexMul kernels which the expander realises as SDFG
 * map-nests. Both dispatchers share the radix helpers and the kernel template.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/integer.h"

namespace sdfg::cuda::tensor::fft2d {

// Extract a compile-time integer from a symbolic expression (expander guarantees constness).
inline int64_t const_int(const symbolic::Expression& e) {
    if (!SymEngine::is_a<SymEngine::Integer>(*e)) {
        throw std::runtime_error("FFT2D dispatcher: expected constant dimension, got " + e->__str__());
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*e).as_int();
}

// Smallest 5-smooth number (factors 2,3,5 only) that is >= n.
inline int64_t next_smooth(int64_t n) {
    for (int64_t x = std::max<int64_t>(n, 1);; ++x) {
        int64_t t = x;
        while (t % 2 == 0) t /= 2;
        while (t % 3 == 0) t /= 3;
        while (t % 5 == 0) t /= 5;
        if (t == 1) return x;
    }
}

// Factor a 5-smooth length into a radix sequence (radix 4 before 2, then 3, then 5).
inline std::vector<int> factor_radices(int64_t n) {
    std::vector<int> r;
    while (n % 4 == 0) {
        r.push_back(4);
        n /= 4;
    }
    while (n % 2 == 0) {
        r.push_back(2);
        n /= 2;
    }
    while (n % 3 == 0) {
        r.push_back(3);
        n /= 3;
    }
    while (n % 5 == 0) {
        r.push_back(5);
        n /= 5;
    }
    return r;
}

inline void replace_all(std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Hand-tuned mixed-radix Stockham FFT kernels operating on native complex (float2)
// buffers. `%P%` is replaced by a per-node prefix so multiple FFT nodes never collide.
// Same bodies as the fused fft_conv dispatcher; only the transform kernels are emitted
// here (pad/crop/complexMul are separate SDFG map-nests).
inline const char* KERNELS_TEMPLATE = R"CU(
template<int R>
__device__ __forceinline__ void %P%_radixDFT(float* vr, float* vi, float sign);

template<>
__device__ __forceinline__ void %P%_radixDFT<2>(float* vr, float* vi, float) {
    float t0r = vr[0] + vr[1], t0i = vi[0] + vi[1];
    float t1r = vr[0] - vr[1], t1i = vi[0] - vi[1];
    vr[0] = t0r; vi[0] = t0i; vr[1] = t1r; vi[1] = t1i;
}
template<>
__device__ __forceinline__ void %P%_radixDFT<3>(float* vr, float* vi, float sign) {
    const float c = -0.5f, s = 0.8660254037844387f;
    float sumr = vr[1] + vr[2], sumi = vi[1] + vi[2];
    float difr = vr[1] - vr[2], difi = vi[1] - vi[2];
    float mr = vr[0] + c * sumr, mi = vi[0] + c * sumi;
    float nr = sign * s * difr, ni = sign * s * difi;
    vr[0] = vr[0] + sumr; vi[0] = vi[0] + sumi;
    vr[1] = mr - ni; vi[1] = mi + nr;
    vr[2] = mr + ni; vi[2] = mi - nr;
}
template<>
__device__ __forceinline__ void %P%_radixDFT<4>(float* vr, float* vi, float sign) {
    float ar = vr[0] + vr[2], ai = vi[0] + vi[2];
    float br = vr[0] - vr[2], bi = vi[0] - vi[2];
    float cr = vr[1] + vr[3], ci = vi[1] + vi[3];
    float dr = vr[1] - vr[3], di = vi[1] - vi[3];
    float rdr = -sign * di, rdi = sign * dr;
    vr[0] = ar + cr; vi[0] = ai + ci;
    vr[1] = br + rdr; vi[1] = bi + rdi;
    vr[2] = ar - cr; vi[2] = ai - ci;
    vr[3] = br - rdr; vi[3] = bi - rdi;
}
template<>
__device__ __forceinline__ void %P%_radixDFT<5>(float* vr, float* vi, float sign) {
    const float c1 = 0.30901699437494742f, s1 = 0.95105651629515357f;
    const float c2 = -0.80901699437494742f, s2 = 0.58778525229247313f;
    float t1r = vr[1] + vr[4], t1i = vi[1] + vi[4];
    float t2r = vr[2] + vr[3], t2i = vi[2] + vi[3];
    float u1r = vr[1] - vr[4], u1i = vi[1] - vi[4];
    float u2r = vr[2] - vr[3], u2i = vi[2] - vi[3];
    float x0r = vr[0], x0i = vi[0];
    vr[0] = x0r + t1r + t2r; vi[0] = x0i + t1i + t2i;
    float car = x0r + c1 * t1r + c2 * t2r, cai = x0i + c1 * t1i + c2 * t2i;
    float cbr = x0r + c2 * t1r + c1 * t2r, cbi = x0i + c2 * t1i + c1 * t2i;
    float dar = s1 * u1r + s2 * u2r, dai = s1 * u1i + s2 * u2i;
    float dbr = s2 * u1r - s1 * u2r, dbi = s2 * u1i - s1 * u2i;
    vr[1] = car - sign * dai; vi[1] = cai + sign * dar;
    vr[4] = car + sign * dai; vi[4] = cai - sign * dar;
    vr[2] = cbr - sign * dbi; vi[2] = cbi + sign * dbr;
    vr[3] = cbr + sign * dbi; vi[3] = cbi - sign * dbr;
}

template<int R>
__device__ __forceinline__ void %P%_stockhamStage(const float* pR, const float* pI, float* qR, float* qI,
                                                   int N, int Ns, int L, int estride, int lstride, float sign) {
    const float PI2 = 6.28318530717958647692f;
    int Nt = N / R;
    int work = Nt * L;
    for (int tt = threadIdx.x; tt < work; tt += blockDim.x) {
        int c = tt % L;
        int j = tt / L;
        float ba = sign * PI2 * (float)(j % Ns) / (float)(Ns * R);
        float c1, s1;
        __sincosf(ba, &s1, &c1);
        float vr[R], vi[R];
        float wr = 1.0f, wi = 0.0f;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int a = (j + r * Nt) * estride + c * lstride;
            float xr = pR[a], xi = pI[a];
            vr[r] = xr * wr - xi * wi;
            vi[r] = xr * wi + xi * wr;
            float nwr = wr * c1 - wi * s1;
            float nwi = wr * s1 + wi * c1;
            wr = nwr; wi = nwi;
        }
        %P%_radixDFT<R>(vr, vi, sign);
        int idxD = (j / Ns) * Ns * R + (j % Ns);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            int a = (idxD + r * Ns) * estride + c * lstride;
            qR[a] = vr[r]; qI[a] = vi[r];
        }
    }
}

__device__ float* %P%_stockhamRun(float* aR, float* aI, float* bR, float* bI,
                                  int N, int L, int estride, int lstride,
                                  const int* radices, int nrad, float sign) {
    float* pR = aR; float* pI = aI;
    float* qR = bR; float* qI = bI;
    int Ns = 1;
    for (int s = 0; s < nrad; ++s) {
        int R = radices[s];
        switch (R) {
            case 2: %P%_stockhamStage<2>(pR, pI, qR, qI, N, Ns, L, estride, lstride, sign); break;
            case 3: %P%_stockhamStage<3>(pR, pI, qR, qI, N, Ns, L, estride, lstride, sign); break;
            case 4: %P%_stockhamStage<4>(pR, pI, qR, qI, N, Ns, L, estride, lstride, sign); break;
            case 5: %P%_stockhamStage<5>(pR, pI, qR, qI, N, Ns, L, estride, lstride, sign); break;
        }
        __syncthreads();
        Ns *= R;
        float* t;
        t = pR; pR = qR; qR = t;
        t = pI; pI = qI; qI = t;
    }
    return pR;
}

// Forward real-to-complex FFT along the row width: real [total_rows x fftW] input
// produces complex [total_rows x halfW] output (halfW = fftW/2+1).
__global__ void %P%_fftRowsR2C(float2* out, const float* in, int total_rows, int fftW, int halfW,
                              const int* radices, int nrad, int linesPerBlock) {
    int line0 = blockIdx.x * linesPerBlock;
    int L = min(linesPerBlock, total_rows - line0);
    if (L <= 0) return;
    extern __shared__ float smem[];
    float* aR = smem; float* aI = aR + L * fftW; float* bR = aI + L * fftW; float* bI = bR + L * fftW;
    size_t g0 = (size_t)line0 * fftW;
    for (int t = threadIdx.x; t < L * fftW; t += blockDim.x) {
        int c = t / fftW, e = t - c * fftW;
        aR[e + c * fftW] = in[g0 + (size_t)c * fftW + e]; aI[e + c * fftW] = 0.0f;
    }
    __syncthreads();
    float* rR = %P%_stockhamRun(aR, aI, bR, bI, fftW, L, 1, fftW, radices, nrad, -1.0f);
    float* rI = (rR == aR) ? aI : bI;
    for (int t = threadIdx.x; t < L * halfW; t += blockDim.x) {
        int c = t / halfW, e = t - c * halfW;
        out[(size_t)(line0 + c) * halfW + e] = make_float2(rR[e + c * fftW], rI[e + c * fftW]);
    }
}

// Inverse complex-to-real FFT along the row width: complex [total_rows x halfW]
// input produces real [total_rows x fftW] output, scaled by 1/fftW. The redundant
// upper half of each row is rebuilt from conjugate symmetry at load time.
__global__ void %P%_fftRowsC2R(float* out, const float2* in, int total_rows, int fftW, int halfW,
                              const int* radices, int nrad, int linesPerBlock) {
    int line0 = blockIdx.x * linesPerBlock;
    int L = min(linesPerBlock, total_rows - line0);
    if (L <= 0) return;
    extern __shared__ float smem[];
    float* aR = smem; float* aI = aR + L * fftW; float* bR = aI + L * fftW; float* bI = bR + L * fftW;
    for (int t = threadIdx.x; t < L * fftW; t += blockDim.x) {
        int c = t / fftW, e = t - c * fftW;
        if (e < halfW) {
            float2 v = in[(size_t)(line0 + c) * halfW + e];
            aR[e + c * fftW] = v.x; aI[e + c * fftW] = v.y;
        } else {
            float2 v = in[(size_t)(line0 + c) * halfW + (fftW - e)];
            aR[e + c * fftW] = v.x; aI[e + c * fftW] = -v.y;
        }
    }
    __syncthreads();
    float* rR = %P%_stockhamRun(aR, aI, bR, bI, fftW, L, 1, fftW, radices, nrad, 1.0f);
    float scale = 1.0f / (float)fftW;
    for (int t = threadIdx.x; t < L * fftW; t += blockDim.x) {
        int c = t / fftW, e = t - c * fftW;
        out[(size_t)(line0 + c) * fftW + e] = rR[e + c * fftW] * scale;
    }
}

__global__ void %P%_fftCols(float2* data, int matrices, int rows, int cols, int activeCols,
                            const int* radices, int nrad, float sign, int doScale, int tile, int ldm) {
    int tilesPerMatrix = (activeCols + tile - 1) / tile;
    int matrix_idx = blockIdx.x / tilesPerMatrix;
    if (matrix_idx >= matrices) return;
    int tileIdx = blockIdx.x % tilesPerMatrix;
    int colStart = tileIdx * tile;
    int L = min(tile, activeCols - colStart);
    extern __shared__ float smem[];
    float* aR = smem; float* aI = aR + rows * ldm; float* bR = aI + rows * ldm; float* bI = bR + rows * ldm;
    size_t base = (size_t)matrix_idx * rows * cols + colStart;
    for (int t = threadIdx.x; t < rows * L; t += blockDim.x) {
        int row = t / L, c = t - row * L;
        float2 v = data[base + (size_t)row * cols + c];
        aR[row * ldm + c] = v.x; aI[row * ldm + c] = v.y;
    }
    __syncthreads();
    float* rR = %P%_stockhamRun(aR, aI, bR, bI, rows, L, ldm, 1, radices, nrad, sign);
    float* rI = (rR == aR) ? aI : bI;
    float scale = doScale ? (1.0f / (float)rows) : 1.0f;
    for (int t = threadIdx.x; t < rows * L; t += blockDim.x) {
        int row = t / L, c = t - row * L;
        data[base + (size_t)row * cols + c] = make_float2(rR[row * ldm + c] * scale, rI[row * ldm + c] * scale);
    }
}
)CU";

// Emit the shared transform kernels + forward declarations for a node with the given
// per-node `prefix`. Declares fftRowsR2C, fftRowsC2R and fftCols in the caller header.
inline void emit_fft2d_kernels(codegen::CodegenOutput& out, const std::string& prefix) {
    out.library_snippet_factory.add_global("#include <cuda.h>");

    out.globals_stream << "__global__ void " << prefix
                       << "_fftRowsR2C(float2*, const float*, int, int, int, const int*, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_fftRowsC2R(float*, const float2*, int, int, int, const int*, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_fftCols(float2*, int, int, int, int, const int*, int, float, int, int, int);" << std::endl;

    std::string kernels = KERNELS_TEMPLATE;
    replace_all(kernels, "%P%", prefix);
    auto& ks = out.library_snippet_factory.require(prefix + "_kernels", "cu", true).stream();
    ks << "#include " << out.library_snippet_factory.header_path().filename() << std::endl << std::endl;
    ks << kernels << std::endl;
}

} // namespace sdfg::cuda::tensor::fft2d
