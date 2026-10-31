/*
 * Parity and timing harness. Runs the golden windows through the generated library and
 * prints one line per window. All numbers are printed as raw float32 bits in hex, so the
 * comparison with Python is exact and needs no printf.
 *
 *   T <counter ticks>                      (TINYML_TIMING, when port_cycles() works)
 *   C <instructions> <counter ticks>       (a loop of known length, to turn ticks into instructions)
 *   W <id> <label> <score> <f0> <f1> ...    (TINYML_PARITY)
 */
#include <stdint.h>

#include "golden_vectors.h"
#include "tinyml_model.h"
#include "tinyml_port.h"

#ifndef TINYML_TIMING
#define TINYML_TIMING 1
#endif
#ifndef TINYML_PARITY
#define TINYML_PARITY 1
#endif

volatile int32_t tinyml_sink;

static void put_hex(uint32_t v)
{
    static const char digits[] = "0123456789abcdef";
    char buf[10];
    for (int i = 0; i < 8; i++) {
        buf[i] = digits[(v >> (28 - 4 * i)) & 0xfu];
    }
    buf[8] = ' ';
    buf[9] = '\0';
    port_write(buf);
}

static void put_dec(uint32_t v)
{
    char buf[12];
    int i = 10;
    buf[11] = '\0';
    buf[10] = ' ';
    do {
        buf[--i] = (char)('0' + v % 10u);
        v /= 10u;
    } while (v != 0u && i > 0);
    port_write(&buf[i]);
}

static uint32_t float_bits(float f)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    return c.u;
}

__attribute__((noinline)) void tinyml_mark_begin(void)
{
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void tinyml_mark_end(void)
{
    __asm__ volatile("" ::: "memory");
}

int harness_main(void)
{
    port_init();
    port_write("TINYML BEGIN\n");
#if TINYML_TIMING
    {
        uint32_t cal_instructions = 0u;
        port_irq_disable();
        const uint32_t cal_ticks = port_calibrate(&cal_instructions);
        const uint32_t c0 = port_cycles();
        tinyml_mark_begin();
        const tinyml_result_t r = model_predict(tinyml_golden_inputs[0]);
        tinyml_mark_end();
        const uint32_t c1 = port_cycles();
        port_irq_enable();
        tinyml_sink = r.label;
        port_write("T ");
        put_dec(c1 - c0);
        port_write("\n");
        if (cal_ticks != 0u) {
            port_write("C ");
            put_dec(cal_instructions);
            put_dec(cal_ticks);
            port_write("\n");
        }
    }
#endif
#if TINYML_PARITY
    for (int g = 0; g < TINYML_GOLDEN_COUNT; g++) {
        float f[TINYML_NUM_FEATURES];
        model_features(tinyml_golden_inputs[g], f);
        const tinyml_result_t r = model_predict(tinyml_golden_inputs[g]);
        port_write("W ");
        put_dec((uint32_t)g);
        put_dec((uint32_t)r.label);
        put_hex(float_bits(r.score));
        for (int i = 0; i < TINYML_NUM_FEATURES; i++) {
            put_hex(float_bits(f[i]));
        }
        port_write("\n");
    }
#endif
    port_write("TINYML DONE\n");
    port_exit(0);
    return 0;
}
