# Floating-Point Math

Floating-point arithmetic is not real arithmetic: every operation rounds its result to the nearest representable value.
Transformations that are trivially correct over the reals — reassociating a sum, contracting a multiply and an add — change *where* that rounding happens and therefore, in general, the bits of the result.
Compilers expose these rewrites behind explicit opt-ins (`-ffp-contract`, `-fassociative-math`, `-ffast-math`, ...).
This module collects the SDFG transformations that rely on such relaxed floating-point semantics.

## Contraction: `-ffp-contract=fast`

A *fused multiply-add* computes `a * b + c` with a single rounding at the end, instead of rounding the product and then the sum:

```
fp_mul + fp_add:  d = round(round(a * b) + c)
fp_fma:           d = round(a * b + c)
```

Most hardware executes an FMA in one instruction, and the result is at least as accurate as the unfused sequence — but it is a *different* result.
Contraction is therefore legal only if the user opts into it; `-ffp-contract=fast` allows it across statements, which is what an SDFG block, holding the dataflow of many statements, needs.

| Transformation | Rewrite | Notes |
|---|---|---|
| `FuseFMA(add)` | `t = a * b; d = t + c` → `d = fma(a, b, c)` | `t` must be a transient scalar written by the `fp_mul` and read only by the `fp_add`, both in the same block; all operands of the same precision. |
| `UnfuseFMA(fma)` | `d = fma(a, b, c)` → `t = a * b; d = t + c` | Inverse of `FuseFMA`; introduces a new transient scalar `t` for the product. |

`FPContractionPass` applies `FuseFMA` in every block until a fixed point is reached.
It runs at the end of the Python frontend's `simplify` if `TargetOptions::fp_contract` (`DoccOptions(fp_contract=...)`) is set, which is the default, so the SDFG is consistent with frontends that already emit `fp_fma` (e.g. clang's default contraction); later passes (einsum detection, reduction recognition, targets) handle `fp_fma` like a multiply/add pair.
With `fp_contract=False` no contraction is performed, while FMAs created explicitly by a frontend are kept.
`UnfuseFMA` is used where a target prefers the split form, e.g. to hand the product on to a separate computation.

## Low-precision contractions: matmuls on integer units

Matrix units are far faster in Int8 than in Float or Double, e.g. 2x FP16 on AMD CDNA3 and ~30x FP64 on NVIDIA H100.
The transformations below compute a Float or Double `MatMul` (`Y = A @ B`) or row-major `GEMM` (`C = alpha A @ B + beta C`, transposes allowed) from *exact* Int8 matmuls — `MatMul` library nodes with `ContractionTypes{Int8, Int32, Int32}` that the target maps, e.g. onto integer matrix units — plus elementwise maps that split the operands and combine the products.
Every row of A and column of B is scaled by its own factor, so rows and columns of different magnitudes keep their precision.

| Transformation | Result | Int8 matmuls | Accuracy |
|---|---|---|---|
| `QuantizeContraction(matmul, bits = 8)` | `Y ≈ (A_q @ B_q) · s_A[i] · s_B[j] / q²` with `A_q = rint(q · A / s_A)`, `q = 2^(bits-1) - 1`, `s` the absmax of the row or column | 1 | lossy: ~1/q of the largest magnitudes (3.6e-3 measured for 8 bits) |
| `OzakiContraction(matmul, Slicing, s)` | Ozaki-I: `A / 2^e = Σ_t 2^-7t A_t` in Int8 slices; `Y = 2^(e_i+f_j) Σ_{t+u ≤ s+1} 2^-7(t+u) A_t @ B_u` | s(s+1)/2 | ~7s bits per row and column; default s = 4 (Float), 8 (Double) |
| `OzakiContraction(matmul, Modular, L)` | Ozaki-II: integers `A' = trunc(2^q A / 2^e)`, their product modulo L coprime moduli (one matmul each), reconstructed with Garner's algorithm | L | q bits with `K 2^2q < Π m_l / 4`; default L reaches 26 (Float) or 55 (Double) bits |

The Ozaki variants are as accurate as the native matmul — 1.7e-16 (Double) and 2.3e-8 (Float) measured against 1.7e-16 and 5.6e-8 natively, relative to `Σ_k |A[i,k] B[k,j]|` — but not bitwise equal to it.
Like every Ozaki scheme they lose bits on elements much smaller than the largest of their row or column.

Where they pay off:
- **Ozaki** where the native precision is much slower than Int8: FP64 on GPUs without fast FP64 (consumer and inference GPUs), or FP32 on integer-only accelerators. With s(s+1)/2 or L matmuls it cannot beat a native unit only twice as slow as Int8.
- **Quantization** for inference, where the error is acceptable, on large, compute-bound matmuls; the absmax and quantize passes cost O(MK + KN) memory traffic.

On AMD GPUs the generated matrix-core path (`GpuMmaExpander`, `GpuMmaEinsumTransform`) maps the Int8 matmuls onto Int8 MFMA/WMMA with Int32 accumulators, using the full-rate 16x16x32 blocks on CDNA3 (gfx942) and 16x16x16 on CDNA2 and RDNA.

Requirements: a 2-D MatMul whose inputs, accumulator and output share the Float or Double type, or a row-major Float or Double GEMM, and a constant K for which the Int32 accumulation is exact (K ≤ 133143 for Int8 values up to ±127).
The splitting and reconstruction run in Double; inputs of magnitude 2^1023 or above are not supported.

### Selecting the precision

`ContractionPrecisionPass` applies one of these transformations to every MatMul and GEMM it can; the others stay native.
The Python frontends select it with `DoccOptions(contraction_precision=...)` (`TargetOptions::contraction_precision`), applied at the start of `expand`, before the targets map matmuls (e.g. to BLAS), so they map the Int8 matmuls instead:

| `contraction_precision` | Computation |
|---|---|
| `native` (default) | unchanged |
| `int8`, `int2` ... `int7` | `QuantizeContraction` with that many bits (lossy) |
| `ozaki` | `OzakiContraction`, modular, which needs the fewest Int8 matmuls |
| `ozaki1[:s]` / `ozaki2[:L]` | `OzakiContraction`, slicing with s slices / modular with L moduli (default: by precision) |

With the frontend's default symbolic shapes K is not a constant, so pass `symbolic_shapes=False` for the transformations to apply.
