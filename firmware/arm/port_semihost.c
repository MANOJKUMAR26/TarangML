/* Arm semihosting port: output and exit through the debugger channel QEMU provides. */
#include "tinyml_port.h"

static int semihost(int op, const void *arg)
{
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xab" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void port_init(void) {}

void port_write(const char *text)
{
    semihost(0x04, text); /* SYS_WRITE0 */
}

void port_exit(int code)
{
    semihost(0x18, (const void *)(code == 0 ? 0x20026 : 0x20023)); /* SYS_EXIT */
    for (;;) {
    }
}

uint32_t port_cycles(void)
{
    return 0u; /* the runner counts instructions from the QEMU execution trace instead */
}

uint32_t port_calibrate(uint32_t *instructions)
{
    *instructions = 0u;
    return 0u;
}

void port_irq_disable(void) {}
void port_irq_enable(void) {}
