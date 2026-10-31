# Parity and timing firmware for Espressif chips. The tinyml runner copies the generated
# library, the golden vectors and the shared harness into main/ before building.
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tinyml_harness)
