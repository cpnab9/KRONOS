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
    "${SOFTWARE_DIR}/Eigen3" # 针对直接 clone 源码不编译的情况
)
link_directories(
    "${SOFTWARE_DIR}/fatrop/lib" 
    "${SOFTWARE_DIR}/fatrop/lib64"
)

find_package(fatrop REQUIRED)

# 尝试寻找 Eigen3 的 CMake 配置。如果不成功，因为上面已经显式包含了 include 目录，
# 我们设为 QUIET（不强制报错），这样直接解压的源码也能工作。
find_package(Eigen3 QUIET) 

# ==========================================
# 2. 编译生成的算例库 (Plugin) - 特殊处理
# ==========================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/nlp_code.c")
    # 把 NLP 和 动力学导数 C 文件一起编译进动态库
    add_library(nlp_case SHARED 
        generated/nlp_code.c
        generated/f_cont_code.c
    )
    target_link_libraries(nlp_case PRIVATE fatrop::fatrop)
    set_target_properties(nlp_case PROPERTIES PREFIX "")
    
    # 【核心】：单独给这两个容易卡死的巨型 C 文件施加 -O0 (不优化)
    set_target_properties(nlp_case PROPERTIES COMPILE_FLAGS "-O0")
    
    message(STATUS "[KRONOS] Found generated NLP and f_cont files, target nlp_case configured.")
else()
    message(WARNING "[KRONOS] generated/nlp_code.c not found.")
endif()

# ==========================================
# 3. 编译通用求解器程序 (Solver Engine)
# ==========================================
add_executable(kronos_solver 
    src/main.cpp
    src/solver/fatrop_wrapper.cpp
)
target_include_directories(kronos_solver PRIVATE include generated)

# 动态链接判断：如果找到了 Eigen3::Eigen target 就链接，否则由于已经是纯头文件包含，则无需链接
if(TARGET Eigen3::Eigen)
    target_link_libraries(kronos_solver PRIVATE fatrop::fatrop Eigen3::Eigen ${CMAKE_DL_LIBS})
else()
    target_link_libraries(kronos_solver PRIVATE fatrop::fatrop ${CMAKE_DL_LIBS})
endif()