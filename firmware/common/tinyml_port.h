/* Board layer used by the parity harness and the size probe. One small file per platform. */
#ifndef TINYML_PORT_H
#define TINYML_PORT_H

#include <stdint.h>

void port_init(void);
void port_write(const char *text);
void port_exit(int code);
uint32_t port_cycles(void); /* instruction/cycle counter if the platform has one, else 0 */
/* Runs a loop of known length; returns the counter ticks it took and sets *instructions. 0 if unsupported. */
uint32_t port_calibrate(uint32_t *instructions);
void port_irq_disable(void);
void port_irq_enable(void);

#endif
