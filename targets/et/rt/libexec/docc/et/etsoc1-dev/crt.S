/*-------------------------------------------------------------------------
 * Minimal CRT startup for ET device kernels.
 * Provides _start, which initializes gp, calls entry_point(),
 * then returns to firmware via ecall.
 *
 * Based on test-compute-kernels/src/shared/crt.S
 *-------------------------------------------------------------------------*/

#include "etsoc/isa/syscall.h"

.section .text.init, "ax", @progbits
.global _start

_start:
    # initialize global pointer
.option push
.option norelax
    la    gp, __global_pointer$
.option pop
    # Firmware sets stack pointer before launch
    # bss not allowed, no init
    call  entry_point

    li    a2, KERNEL_RETURN_SUCCESS
    mv    a1, a0
    li    a0, SYSCALL_RETURN_FROM_KERNEL
    ecall
