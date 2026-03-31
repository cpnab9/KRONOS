cmake_minimum_required(VERSION 3.10)
project(KRONOS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加三个头文件搜索路径
include_directories(
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/generated
    ${CMAKE_SOURCE_DIR}/include
)

# 编译所有组件
add_executable(kronos 
    src/main.cpp 
    src/kronos_nlp_wrapper.cpp
    src/kronos_solver.cpp
    generated/kkt_funcs.c
)