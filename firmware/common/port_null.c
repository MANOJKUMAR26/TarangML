/* Port with no I/O, used only to link size probes. */
#include "tinyml_port.h"

void port_init(void) {}
void port_write(const char *text) { (void)text; }
void port_exit(int code)
{
    (void)code;
    for (;;) {
    }
}
uint32_t port_cycles(void) { return 0u; }
uint32_t port_calibrate(uint32_t *instructions)
{
    *instructions = 0u;
    return 0u;
}
void port_irq_disable(void) {}
void port_irq_enable(void) {}
