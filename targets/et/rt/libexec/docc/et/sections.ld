/*-------------------------------------------------------------------------
 * Linker script for ET device kernels.
 * Based on test-compute-kernels/src/shared/sections.ld
 *-------------------------------------------------------------------------*/

/* Entry point of a user kernel
   WARNING: Fixed address (must be synced with device's memory layout) */
KERNEL_UMODE_ENTRY = 0x8005801000;

ENTRY(_start)

SECTIONS
{
    . = KERNEL_UMODE_ENTRY;

    .text :
    {
        /* Place boot code first */
        *(.text.init)
        *(SORT_BY_ALIGNMENT(.text) SORT_BY_ALIGNMENT(.text.*) SORT_BY_ALIGNMENT(.gnu.linkonce.t.*))
    }

    .rodata :
    {
        *(SORT_BY_ALIGNMENT(.rodata) SORT_BY_ALIGNMENT(.rodata.*) SORT_BY_ALIGNMENT(.gnu.linkonce.r.*))
    }

    .rodata1 :
    {
        *(SORT_BY_ALIGNMENT(.rodata1))
    }

    .sdata2 :
    {
        *(SORT_BY_ALIGNMENT(.sdata2) SORT_BY_ALIGNMENT(.sdata2.*) SORT_BY_ALIGNMENT(.gnu.linkonce.s2.*))
    }

    .sbss2 :
    {
        *(SORT_BY_ALIGNMENT(.sbss2) SORT_BY_ALIGNMENT(.sbss2.*) SORT_BY_ALIGNMENT(.gnu.linkonce.sb2.*))
    }

    .data :
    {
        *(.data .data.* .gnu.linkonce.d.*)
    }

    .sdata :
    {
        __global_pointer$ = . + 0x800;
        *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
        *(.sdata .sdata.* .gnu.linkonce.s.*)
    }

    .bss :
    {
        . = ALIGN(4);
        __bss_start = .;
        *(.bss .bss.* .gnu.linkonce.b.*)
    }

    .sbss :
    {
        *(.sbss .sbss.* .gnu.linkonce.sb.*)
        . = ALIGN(4);
        __bss_end = .;
    }

    /DISCARD/ : { *(.riscv.attributes) }
}

ASSERT((__bss_start == __bss_end), "bss not allowed in kernels");
