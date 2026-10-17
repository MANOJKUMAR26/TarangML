/*
 * Size probe: the smallest program that keeps the whole model alive. Built twice, with and
 * without TINYML_PROBE_EMPTY; the difference in the linker output is what the model costs.
 */
#include <stdint.h>

#include "tinyml_port.h"
#ifndef TINYML_PROBE_EMPTY
#include "tinyml_model.h"
static float tinyml_probe_window[TINYML_WINDOW_SAMPLES * TINYML_NUM_AXES];
#endif

volatile int32_t tinyml_probe_sink;

int harness_main(void)
{
    port_init();
#ifndef TINYML_PROBE_EMPTY
    const tinyml_result_t r = model_predict(tinyml_probe_window);
    tinyml_probe_sink = r.label;
#endif
    port_exit(0);
    return 0;
}
