/* RISC-V semihosting port (the slli/ebreak/srai sequence QEMU recognises). */
#include "tinyml_port.h"

static int semihost(int op, const void *arg)
{
    register long a0 __asm__("a0") = op;
    register const void *a1 __asm__("a1") = arg;
    __asm__ volatile(
        ".option push\n\t"
        ".option norvc\n\t"
        ".balign 16\n\t"
        "slli x0, x0, 0x1f\n\t"
        "ebreak\n\t"
        "srai x0, x0, 7\n\t"
        ".option pop"
        : "+r"(a0)
        : "r"(a1)
        : "memory");
    return (int)a0;
}

void port_init(void) {}

void port_write(const char *text)
{
    semihost(0x04, text); /* SYS_WRITE0 */
}

void port_exit(int code)
{
    semihost(0x18, (const void *)(long)(code == 0 ? 0x20026 : 0x20023)); /* SYS_EXIT */
    for (;;) {
    }
}

uint32_t port_cycles(void)
{
    return 0u;
}

uint32_t port_calibrate(uint32_t *instructions)
{
    *instructions = 0u;
    return 0u;
}

void port_irq_disable(void) {}
void port_irq_enable(void) {}
