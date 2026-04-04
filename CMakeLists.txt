cmake_minimum_required(VERSION 3.10)
project(KRONOS_Trajectory_Optimizer)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 全局开启 C++ 的 Release 模式和极限优化
# ==========================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math")

# ==========================================
# 1. 环境与依赖配置 (并行相对路径管理)
# ==========================================
# 自动定位到 kronos 的上一级目录下的 software 文件夹
set(SOFTWARE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../software")

# 将 Eigen3 和 fatrop 的路径加入 CMake 查找路径
list(APPEND CMAKE_PREFIX_PATH "${SOFTWARE_DIR}/fatrop" "${SOFTWARE_DIR}/Eigen3")

# 增加防御性包含：即使库只是直接解压没经过 cmake install，也能找到头文件和库
include_directories(
    "${SOFTWARE_DIR}/fatrop/include"
    "${SOFTWARE_DIR}/Eigen3/include/eigen3"
    "${SOFTWARE_DIR}/Eigen3" 
)
link_directories(
    "${SOFTWARE_DIR}/fatrop/lib" 
    "${SOFTWARE_DIR}/fatrop/lib64"
)

find_package(fatrop REQUIRED)
find_package(Eigen3 QUIET) 

# ==========================================
# 2. 编译生成的算例库 (Plugin) - 动态捕获
# ==========================================
# 【核心修改】：不再硬编码文件名，而是抓取 generated 目录下的所有 .c 文件
file(GLOB GENERATED_C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/generated/*.c")

if(GENERATED_C_FILES)
    # 将所有找到的 C 文件（nlp_code.c, f_cont_code.c, metadata.c 等）打包成 kronos_problem 动态库
    add_library(kronos_problem SHARED ${GENERATED_C_FILES})
    
    target_link_libraries(kronos_problem PRIVATE fatrop::fatrop)
    
    # 保持前缀为空，以便生成 libkronos_problem.so (或者在某些系统上直接是 kronos_problem.so)
    set_target_properties(kronos_problem PROPERTIES PREFIX "")
    
    # 【重要】：单独给自动生成的巨型 C 文件施加 -O0 避免编译器内存溢出/卡死
    set_target_properties(kronos_problem PROPERTIES COMPILE_FLAGS "-O0")
    
    message(STATUS "[KRONOS] Found generated problem files, target 'kronos_problem' configured.")
else()
    message(WARNING "[KRONOS] No C files found in generated/ directory. Problem library will not be built.")
endif()

# ==========================================
# 3. 编译通用求解器程序 (Solver Engine)
# ==========================================
add_executable(kronos_solver 
    src/main.cpp
    src/solver/fatrop_wrapper.cpp
)
target_include_directories(kronos_solver PRIVATE include generated)

# 动态链接判断：如果找到了 Eigen3::Eigen target 就链接
if(TARGET Eigen3::Eigen)
    target_link_libraries(kronos_solver PRIVATE fatrop::fatrop Eigen3::Eigen ${CMAKE_DL_LIBS})
else()
    target_link_libraries(kronos_solver PRIVATE fatrop::fatrop ${CMAKE_DL_LIBS})
endif()