/* Minimal Cortex-M start-up: copy .data, clear .bss, enable the FPU, run the harness. */
#include <stdint.h>

#include "tinyml_port.h"

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int harness_main(void);

void Reset_Handler(void)
{
    uint32_t *src = &_sidata;
    for (uint32_t *dst = &_sdata; dst < &_edata;) {
        *dst++ = *src++;
    }
    for (uint32_t *dst = &_sbss; dst < &_ebss;) {
        *dst++ = 0u;
    }
#if defined(__ARM_FP)
    *(volatile uint32_t *)0xE000ED88u |= (0xFu << 20); /* CPACR: full access to CP10/CP11 */
    __asm__ volatile("dsb\n\tisb" ::: "memory");
#endif
    harness_main();
    port_exit(0);
    for (;;) {
    }
}

void Fault_Handler(void)
{
    port_write("TINYML FAULT\n");
    port_exit(3);
    for (;;) {
    }
}

__attribute__((section(".isr_vector"), used)) static void (*const vectors[16])(void) = {
    (void (*)(void))(&_estack), Reset_Handler, Fault_Handler, Fault_Handler, Fault_Handler, Fault_Handler,
    Fault_Handler, 0, 0, 0, 0, Fault_Handler, Fault_Handler, 0, Fault_Handler, Fault_Handler,
};
