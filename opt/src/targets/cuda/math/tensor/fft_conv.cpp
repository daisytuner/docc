#include "sdfg/targets/cuda/math/tensor/fft_conv.h"

#include <optional>
#include <string>
#include <vector>

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "symengine/integer.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::cuda::tensor {

using math::tensor::FFTConvNode;

namespace {

// Extract a compile-time integer from a symbolic expression (expander guarantees constness).
int64_t const_int(const symbolic::Expression& e) {
    if (!SymEngine::is_a<SymEngine::Integer>(*e)) {
        throw std::runtime_error("FFTConv dispatcher: expected constant dimension, got " + e->__str__());
    }
    return SymEngine::down_cast<const SymEngine::Integer&>(*e).as_int();
}

// Smallest 5-smooth number (factors 2,3,5 only) that is >= n.
int64_t next_smooth(int64_t n) {
    for (int64_t x = std::max<int64_t>(n, 1);; ++x) {
        int64_t t = x;
        while (t % 2 == 0) t /= 2;
        while (t % 3 == 0) t /= 3;
        while (t % 5 == 0) t /= 5;
        if (t == 1) return x;
    }
}

// Factor a 5-smooth length into a radix sequence (radix 4 before 2, then 3, then 5).
std::vector<int> factor_radices(int64_t n) {
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

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Hand-tuned mixed-radix Stockham FFT kernels operating on native complex (float2)
// buffers. `%P%` is replaced by a per-node prefix so multiple FFTConv nodes never
// collide. Mirrors /home/adrian/fft_conv_tuned.cu, but with a single complex buffer
// instead of separate real/imag arrays.
const char* KERNELS_TEMPLATE = R"CU(
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
// produces complex [total_rows x halfW] output (halfW = fftW/2+1). The redundant
// upper half of each row's Hermitian spectrum is never materialised.
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
// upper half of each row is rebuilt on the fly from conjugate symmetry at load time.
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

// Pointwise complex multiply over the half-width spectrum [., fftH, halfW]. Both
// operands are stored half-width (R2C), so no full-width remapping is needed; the
// kernel spectrum is broadcast across the batch (channel-indexed).
__global__ void %P%_complexMul(float2* a, const float2* b, int batch, int channels, int fftH, int halfW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int per = fftH * halfW;
    int total = batch * channels * per;
    if (idx >= total) return;
    int channel_idx = (idx / per) % channels;
    int w_idx = channel_idx * per + (idx % per);
    float2 x = a[idx]; float2 w = b[w_idx];
    a[idx] = make_float2(x.x * w.x - x.y * w.y, x.x * w.y + x.y * w.x);
}

__global__ void %P%_pad(float* dst, const float* src, int batch, int channels, int inH, int inW,
                        int fftH, int fftW, int shiftH, int shiftW, int flip) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * fftH * fftW;
    if (idx >= total) return;
    int w = idx % fftW;
    int r = (idx / fftW) % fftH;
    int c = (idx / (fftW * fftH)) % channels;
    int b = idx / (fftW * fftH * channels);
    float val = 0.0f;
    if (r >= shiftH && r < shiftH + inH && w >= shiftW && w < shiftW + inW) {
        int sr = r - shiftH, sw = w - shiftW;
        if (flip) { sr = inH - 1 - sr; sw = inW - 1 - sw; }
        val = src[b * (channels * inH * inW) + c * (inH * inW) + sr * inW + sw];
    }
    dst[idx] = val;
}

__global__ void %P%_crop(float* dst, const float* src, const float* bias,
                         int batch, int channels, int inH, int inW, int fftH, int fftW,
                         int shiftH, int shiftW, int hasBias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * inH * inW;
    if (idx >= total) return;
    int w = idx % inW;
    int r = (idx / inW) % inH;
    int c = (idx / (inW * inH)) % channels;
    int b = idx / (inW * inH * channels);
    int srcIdx = b * (channels * fftH * fftW) + c * (fftH * fftW) + (r + shiftH) * fftW + (w + shiftW);
    float v = src[srcIdx];
    if (hasBias) v += bias[c];
    dst[idx] = v;
}
)CU";

// Shared emitter for both the with- and without-transfers dispatchers. When
// `with_transfers` is true the operands (X, W, bias, Y) are host buffers: the emitter
// allocates matching device buffers and copies host<->device around the kernels. When
// false the operands are already device-resident (data-transfer extraction has pulled
// the copies out into explicit offloading nodes) and are used directly; only the
// internal scratch buffers (spectra + radix tables) are managed here.
void emit_fft_conv(
    codegen::CodegenOutput& out,
    codegen::LanguageExtension& lang,
    const FFTConvNode& node,
    std::vector<codegen::DispatchInput>& inputs,
    bool with_transfers
) {
    if (node.real_primitive() != types::PrimitiveType::Float) {
        throw std::runtime_error("FFTConv CUDA dispatcher (v1) supports single precision only");
    }

    // Geometry (constant; guaranteed by the expander).
    const int64_t N = const_int(node.shape()[0]);
    const int64_t C = const_int(node.shape()[1]);
    const int64_t H = const_int(node.shape()[2]);
    const int64_t W = const_int(node.shape()[3]);
    const int64_t Kh = const_int(node.kernel_shape()[0]);
    const int64_t Kw = const_int(node.kernel_shape()[1]);
    const int64_t pad_h = const_int(node.pads()[0]);
    const int64_t pad_w = const_int(node.pads()[1]);

    const int64_t fftH = next_smooth(H + Kh - 1);
    const int64_t fftW = next_smooth(W + Kw - 1);
    const auto radW = factor_radices(fftW);
    const auto radH = factor_radices(fftH);

    // Connector pointer expressions (order follows the node's input list).
    const std::string y_ptr = inputs.at(FFTConvNode::Y_INPUT_IDX).expr;
    const std::string x_ptr = inputs.at(FFTConvNode::X_INPUT_IDX).expr;
    const std::string w_ptr = inputs.at(FFTConvNode::W_INPUT_IDX).expr;
    const bool has_bias = node.with_bias();
    const std::string b_ptr = has_bias ? inputs.at(FFTConvNode::B_INPUT_IDX).expr : std::string("nullptr");

    // Operand buffers: locally-allocated device copies (with transfers) or the
    // already-device-resident operands passed straight through (without transfers).
    const std::string dX = with_transfers ? std::string("__fc_dX") : x_ptr;
    const std::string dW = with_transfers ? std::string("__fc_dW") : w_ptr;
    const std::string dOut = with_transfers ? std::string("__fc_dOut") : y_ptr;
    const std::string dBias = has_bias ? (with_transfers ? std::string("__fc_dBias") : b_ptr) : std::string("nullptr");

    const std::string prefix = "__fftconv_" + std::to_string(node.element_id());

    out.library_snippet_factory.add_global("#include <cuda.h>");

    // Forward-declare the launched kernels in the header so the caller TU sees them.
    out.globals_stream << "__global__ void " << prefix
                       << "_fftRowsR2C(float2*, const float*, int, int, int, const int*, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_fftRowsC2R(float*, const float2*, int, int, int, const int*, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_fftCols(float2*, int, int, int, int, const int*, int, float, int, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix << "_complexMul(float2*, const float2*, int, int, int, int);"
                       << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_pad(float*, const float*, int, int, int, int, int, int, int, int, int);" << std::endl;
    out.globals_stream << "__global__ void " << prefix
                       << "_crop(float*, const float*, const float*, int, int, int, int, int, int, int, int, int);"
                       << std::endl;

    // Emit the kernel bodies into a per-node .cu snippet.
    {
        std::string kernels = KERNELS_TEMPLATE;
        replace_all(kernels, "%P%", prefix);
        auto& ks = out.library_snippet_factory.require(prefix + "_kernels", "cu", true).stream();
        ks << "#include " << out.library_snippet_factory.header_path().filename() << std::endl << std::endl;
        ks << kernels << std::endl;
    }

    // ---- Host launch sequence ------------------------------------------------------------
    auto& s = out.stream;
    s << "{" << std::endl;
    s.setIndent(s.indent() + 4);

    // Constant geometry.
    s << "const int __fc_N = " << N << ", __fc_C = " << C << ", __fc_H = " << H << ", __fc_W = " << W << ";"
      << std::endl;
    s << "const int __fc_Kh = " << Kh << ", __fc_Kw = " << Kw << ";" << std::endl;
    s << "const int __fc_fftH = " << fftH << ", __fc_fftW = " << fftW << ";" << std::endl;
    // Real input => Hermitian spectrum; only the first fftW/2+1 columns are unique (R2C/C2R).
    s << "const int __fc_halfW = __fc_fftW / 2 + 1;" << std::endl;
    // Operands are placed at the spectral origin (shift 0); the "same"-padding output
    // window is recovered by the crop offset below.
    s << "const int __fc_shiftH = 0, __fc_shiftW = 0;" << std::endl;
    s << "const int __fc_spatial = __fc_N * __fc_C * __fc_H * __fc_W;" << std::endl;
    // Real padded buffers are full width; the complex spectra keep only halfW columns.
    s << "const int __fc_pad_img_elems = __fc_N * __fc_C * __fc_fftH * __fc_fftW;" << std::endl;
    s << "const int __fc_pad_ker_elems = __fc_C * __fc_fftH * __fc_fftW;" << std::endl;
    s << "const int __fc_spec_img_elems = __fc_N * __fc_C * __fc_fftH * __fc_halfW;" << std::endl;
    s << "const int __fc_spec_ker_elems = __fc_C * __fc_fftH * __fc_halfW;" << std::endl;
    s << "const int __fc_weight_elems = __fc_C * __fc_Kh * __fc_Kw;" << std::endl;

    // Radix tables.
    auto emit_radix = [&](const std::string& name, const std::vector<int>& r) {
        s << "int " << name << "[" << r.size() << "] = {";
        for (size_t i = 0; i < r.size(); ++i) {
            s << r[i] << (i + 1 < r.size() ? ", " : "");
        }
        s << "};" << std::endl;
    };
    emit_radix("__fc_hRadW", radW);
    emit_radix("__fc_hRadH", radH);
    s << "const int __fc_nRadW = " << radW.size() << ", __fc_nRadH = " << radH.size() << ";" << std::endl;

    s << "cudaError_t err_cuda;" << std::endl;
    if (with_transfers) {
        s << "float *__fc_dX, *__fc_dW, *__fc_dBias = nullptr, *__fc_dOut;" << std::endl;
    }
    s << "float *__fc_dPadImg, *__fc_dPadKer;" << std::endl;
    s << "float2 *__fc_dImg, *__fc_dKer;" << std::endl;
    s << "int *__fc_dRadW, *__fc_dRadH;" << std::endl;

    auto malloc_check = [&](const std::string& var, const std::string& bytes, const std::string& type) {
        s << "err_cuda = cudaMalloc((void**) &" << var << ", " << bytes << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
        (void) type;
    };
    if (with_transfers) {
        malloc_check("__fc_dX", "__fc_spatial * sizeof(float)", "float");
        malloc_check("__fc_dW", "__fc_weight_elems * sizeof(float)", "float");
        malloc_check("__fc_dOut", "__fc_spatial * sizeof(float)", "float");
    }
    malloc_check("__fc_dPadImg", "__fc_pad_img_elems * sizeof(float)", "float");
    malloc_check("__fc_dPadKer", "__fc_pad_ker_elems * sizeof(float)", "float");
    malloc_check("__fc_dImg", "__fc_spec_img_elems * sizeof(float2)", "float2");
    malloc_check("__fc_dKer", "__fc_spec_ker_elems * sizeof(float2)", "float2");
    malloc_check("__fc_dRadW", "__fc_nRadW * sizeof(int)", "int");
    malloc_check("__fc_dRadH", "__fc_nRadH * sizeof(int)", "int");
    if (with_transfers && has_bias) {
        malloc_check("__fc_dBias", "__fc_C * sizeof(float)", "float");
    }

    auto h2d = [&](const std::string& dst, const std::string& src, const std::string& bytes) {
        s << "err_cuda = cudaMemcpy(" << dst << ", " << src << ", " << bytes << ", cudaMemcpyHostToDevice);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    };
    if (with_transfers) {
        h2d("__fc_dX", x_ptr, "__fc_spatial * sizeof(float)");
        h2d("__fc_dW", w_ptr, "__fc_weight_elems * sizeof(float)");
    }
    h2d("__fc_dRadW", "__fc_hRadW", "__fc_nRadW * sizeof(int)");
    h2d("__fc_dRadH", "__fc_hRadH", "__fc_nRadH * sizeof(int)");
    if (with_transfers && has_bias) {
        h2d("__fc_dBias", b_ptr, "__fc_C * sizeof(float)");
    }

    // Launch geometry (mirrors the reference heuristics).
    s << "int __fc_threads = 256;" << std::endl;
    s << "int __fc_rowLines = 16;" << std::endl;
    s << "while (__fc_rowLines > 1 && (size_t)4 * __fc_rowLines * __fc_fftW * sizeof(float) > 32768) __fc_rowLines >>= "
         "1;"
      << std::endl;
    s << "size_t __fc_rowSmem = (size_t)4 * __fc_rowLines * __fc_fftW * sizeof(float);" << std::endl;
    s << "int __fc_colTile = 32;" << std::endl;
    s << "while (__fc_colTile > 4 && (size_t)4 * __fc_fftH * __fc_colTile * sizeof(float) > 32768) __fc_colTile >>= 1;"
      << std::endl;
    // Bank-conflict guard: pad the shared-memory leading dimension only when the spectrum
    // is narrower than one column tile (single partial tile, L < tile), where consecutive
    // rows share banks. With full tiles (L == tile) the access is already contiguous, so
    // padding would only cost occupancy -> keep ldm = tile there.
    s << "int __fc_colLdm = (__fc_halfW < __fc_colTile) ? __fc_colTile + 1 : __fc_colTile;" << std::endl;
    s << "size_t __fc_colSmem = (size_t)4 * __fc_fftH * __fc_colLdm * sizeof(float);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftRowsR2C, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__fc_rowSmem);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftRowsC2R, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__fc_rowSmem);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftCols, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__fc_colSmem);" << std::endl;

    s << "int __fc_img_rows = __fc_N * __fc_C * __fc_fftH;" << std::endl;
    s << "int __fc_ker_rows = __fc_C * __fc_fftH;" << std::endl;
    s << "int __fc_img_row_blocks = (__fc_img_rows + __fc_rowLines - 1) / __fc_rowLines;" << std::endl;
    s << "int __fc_ker_row_blocks = (__fc_ker_rows + __fc_rowLines - 1) / __fc_rowLines;" << std::endl;
    s << "int __fc_colTilesHalf = (__fc_halfW + __fc_colTile - 1) / __fc_colTile;" << std::endl;
    // Every pass after the forward R2C row transform operates on the half spectrum.
    s << "int __fc_img_half_col_blocks = __fc_N * __fc_C * __fc_colTilesHalf;" << std::endl;
    s << "int __fc_ker_half_col_blocks = __fc_C * __fc_colTilesHalf;" << std::endl;
    s << "int __fc_mul_half_blocks = (__fc_spec_img_elems + __fc_threads - 1) / __fc_threads;" << std::endl;
    s << "int __fc_img_pad_blocks = (__fc_pad_img_elems + __fc_threads - 1) / __fc_threads;" << std::endl;
    s << "int __fc_ker_pad_blocks = (__fc_pad_ker_elems + __fc_threads - 1) / __fc_threads;" << std::endl;
    s << "int __fc_spatial_blocks = (__fc_spatial + __fc_threads - 1) / __fc_threads;" << std::endl;

    // Weight FFT (pad+flip -> R2C rows -> half-width cols).
    s << prefix << "_pad<<<__fc_ker_pad_blocks, __fc_threads>>>(__fc_dPadKer, " << dW
      << ", 1, __fc_C, __fc_Kh, __fc_Kw, "
      << "__fc_fftH, __fc_fftW, __fc_shiftH, __fc_shiftW, 1);" << std::endl;
    s << prefix
      << "_fftRowsR2C<<<__fc_ker_row_blocks, __fc_threads, __fc_rowSmem>>>(__fc_dKer, __fc_dPadKer, __fc_ker_rows, "
      << "__fc_fftW, __fc_halfW, __fc_dRadW, __fc_nRadW, __fc_rowLines);" << std::endl;
    s << prefix << "_fftCols<<<__fc_ker_half_col_blocks, __fc_threads, __fc_colSmem>>>(__fc_dKer, __fc_C, __fc_fftH, "
      << "__fc_halfW, __fc_halfW, __fc_dRadH, __fc_nRadH, -1.0f, 0, __fc_colTile, __fc_colLdm);" << std::endl;

    // Image FFT (pad -> R2C rows -> half-width cols).
    s << prefix << "_pad<<<__fc_img_pad_blocks, __fc_threads>>>(__fc_dPadImg, " << dX
      << ", __fc_N, __fc_C, __fc_H, __fc_W, "
      << "__fc_fftH, __fc_fftW, __fc_shiftH, __fc_shiftW, 0);" << std::endl;
    s << prefix
      << "_fftRowsR2C<<<__fc_img_row_blocks, __fc_threads, __fc_rowSmem>>>(__fc_dImg, __fc_dPadImg, __fc_img_rows, "
      << "__fc_fftW, __fc_halfW, __fc_dRadW, __fc_nRadW, __fc_rowLines);" << std::endl;
    s << prefix
      << "_fftCols<<<__fc_img_half_col_blocks, __fc_threads, __fc_colSmem>>>(__fc_dImg, __fc_N * __fc_C, __fc_fftH, "
      << "__fc_halfW, __fc_halfW, __fc_dRadH, __fc_nRadH, -1.0f, 0, __fc_colTile, __fc_colLdm);" << std::endl;

    // Pointwise complex multiply over the half spectrum.
    s << prefix << "_complexMul<<<__fc_mul_half_blocks, __fc_threads>>>(__fc_dImg, __fc_dKer, __fc_N, __fc_C, "
      << "__fc_fftH, __fc_halfW);" << std::endl;

    // Inverse FFT: half-width cols, then C2R rows write the real result buffer.
    s << prefix
      << "_fftCols<<<__fc_img_half_col_blocks, __fc_threads, __fc_colSmem>>>(__fc_dImg, __fc_N * __fc_C, __fc_fftH, "
      << "__fc_halfW, __fc_halfW, __fc_dRadH, __fc_nRadH, 1.0f, 1, __fc_colTile, __fc_colLdm);" << std::endl;
    s << prefix
      << "_fftRowsC2R<<<__fc_img_row_blocks, __fc_threads, __fc_rowSmem>>>(__fc_dPadImg, __fc_dImg, __fc_img_rows, "
      << "__fc_fftW, __fc_halfW, __fc_dRadW, __fc_nRadW, __fc_rowLines);" << std::endl;

    // Crop + bias. Cross-correlation (torch Conv2d): the kernel is flipped in the
    // spectral domain and both operands sit at the origin, so the valid "same"-padding
    // output window starts at (K - 1 - pad).
    s << "const int __fc_cropH = " << (Kh - 1 - pad_h) << ", __fc_cropW = " << (Kw - 1 - pad_w) << ";" << std::endl;
    s << prefix << "_crop<<<__fc_spatial_blocks, __fc_threads>>>(" << dOut << ", __fc_dPadImg, " << dBias
      << ", __fc_N, __fc_C, "
      << "__fc_H, __fc_W, __fc_fftH, __fc_fftW, __fc_cropH, __fc_cropW, " << (has_bias ? 1 : 0) << ");" << std::endl;
    check_cuda_kernel_launch_errors(s, lang, false);

    // Copy result back to the host operand (only when this dispatcher owns the transfers).
    if (with_transfers) {
        s << "err_cuda = cudaMemcpy(" << y_ptr << ", __fc_dOut, __fc_spatial * sizeof(float), cudaMemcpyDeviceToHost);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }

    for (const std::string& var :
         {"__fc_dPadImg", "__fc_dPadKer", "__fc_dImg", "__fc_dKer", "__fc_dRadW", "__fc_dRadH"}) {
        s << "err_cuda = cudaFree(" << var << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }
    if (with_transfers) {
        for (const std::string& var : {"__fc_dX", "__fc_dW", "__fc_dOut"}) {
            s << "err_cuda = cudaFree(" << var << ");" << std::endl;
            cuda_error_checking(s, lang, "err_cuda");
        }
        if (has_bias) {
            s << "err_cuda = cudaFree(__fc_dBias);" << std::endl;
            cuda_error_checking(s, lang, "err_cuda");
        }
    }

    s.setIndent(s.indent() - 4);
    s << "}" << std::endl;
}

} // namespace

FFTConvNodeDispatcher_CUDAWithTransfers::FFTConvNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTConvNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTConvNodeDispatcher_CUDAWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_fft_conv(out, this->language_extension_, static_cast<const FFTConvNode&>(this->node_), inputs, true);
}

FFTConvNodeDispatcher_CUDAWithoutTransfers::FFTConvNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTConvNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTConvNodeDispatcher_CUDAWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_fft_conv(out, this->language_extension_, static_cast<const FFTConvNode&>(this->node_), inputs, false);
}

} // namespace sdfg::cuda::tensor
