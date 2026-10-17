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
