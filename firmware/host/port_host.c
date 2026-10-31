/* Host port: lets the generated library and harness run natively for a quick first check. */
#include <stdio.h>
#include <stdlib.h>

#include "tinyml_port.h"

int harness_main(void);

void port_init(void) {}

void port_write(const char *text)
{
    fputs(text, stdout);
}

void port_exit(int code)
{
    fflush(stdout);
    exit(code);
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

int main(void)
{
    return harness_main();
}
