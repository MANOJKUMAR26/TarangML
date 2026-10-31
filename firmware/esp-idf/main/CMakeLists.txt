idf_component_register(SRCS "tinyml_model.c" "harness.c" "port_esp.c"
                       INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -fno-math-errno
                       -fstack-usage -fcallgraph-info=su -Wno-unused-function)
