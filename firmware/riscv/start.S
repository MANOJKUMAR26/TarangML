/* Minimal RISC-V start-up for the QEMU virt board. */
    .section .text.start, "ax"
    .global _start
_start:
    .option push
    .option norelax
    la gp, __global_pointer$
    .option pop
    la sp, _estack
    la t0, _sbss
    la t1, _ebss
1:
    bgeu t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:
    call harness_main
    li a0, 0
    call port_exit
3:
    j 3b
