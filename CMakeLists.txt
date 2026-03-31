cmake_minimum_required(VERSION 3.10)
project(KRONOS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/generated
    ${CMAKE_SOURCE_DIR}/include
)

add_executable(kronos 
    src/main.cpp 
    src/kronos_nlp_wrapper.cpp
    src/kronos_kkt_solver.cpp
    src/kronos_optimizer.cpp
    generated/kkt_funcs.c
)