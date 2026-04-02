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
# 1. 环境与依赖配置 (WSL/Linux)
# ==========================================
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
link_directories("$ENV{HOME}/.local/lib")
find_package(fatrop REQUIRED)

# ==========================================
# 2. 编译生成的算例库 (Plugin) - 特殊处理
# ==========================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/nlp_code.c")
    add_library(nlp_case SHARED generated/nlp_code.c)
    target_link_libraries(nlp_case PRIVATE fatrop::fatrop)
    set_target_properties(nlp_case PROPERTIES PREFIX "")
    
    # 【核心】：单独给这一个容易卡死的巨型 C 文件施加 -O0 (不优化)
    set_target_properties(nlp_case PROPERTIES COMPILE_FLAGS "-O0")
    
    message(STATUS "[KRONOS] Found nlp_code.c, target nlp_case configured (Forced -O0 to prevent freezing).")
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
target_include_directories(kronos_solver PRIVATE include)
target_link_libraries(kronos_solver PRIVATE fatrop::fatrop ${CMAKE_DL_LIBS})