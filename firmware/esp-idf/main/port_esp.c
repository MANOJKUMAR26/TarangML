/* ESP-IDF port: UART output, the CPU cycle counter, and app_main as the entry point. */
#include <stdio.h>

#include "esp_cpu.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tinyml_port.h"

int harness_main(void);

void port_init(void) {}

void port_write(const char *text)
{
    fputs(text, stdout);
}

void port_exit(int code)
{
    printf("TINYML EXIT %d\n", code);
    fflush(stdout);
    for (;;) {
        vTaskDelay(1000 / portTICK_PERIOD_MS);
    }
}

uint32_t port_cycles(void)
{
    return (uint32_t)esp_cpu_get_cycle_count();
}

/*
 * In QEMU with -icount the cycle counter advances with virtual time, not one tick per
 * instruction, so the harness also times this loop of exactly 40031 instructions.
 */
uint32_t port_calibrate(uint32_t *instructions)
{
    const uint32_t c0 = (uint32_t)esp_cpu_get_cycle_count();
#if defined(__XTENSA__)
    __asm__ volatile(
        "movi a9, 10\n"
        "2:\n"
        "movi a8, 2000\n"
        "1:\n"
        "addi a8, a8, -1\n"
        "bnez a8, 1b\n"
        "addi a9, a9, -1\n"
        "bnez a9, 2b\n" ::: "a8", "a9");
#else
    __asm__ volatile(
        "li t1, 10\n"
        "2:\n"
        "li t0, 2000\n"
        "1:\n"
        "addi t0, t0, -1\n"
        "bnez t0, 1b\n"
        "addi t1, t1, -1\n"
        "bnez t1, 2b\n" ::: "t0", "t1");
#endif
    const uint32_t c1 = (uint32_t)esp_cpu_get_cycle_count();
    *instructions = 40031u;
    return c1 - c0;
}

void port_irq_disable(void)
{
    portDISABLE_INTERRUPTS();
}

void port_irq_enable(void)
{
    portENABLE_INTERRUPTS();
}

void app_main(void)
{
    harness_main();
}
