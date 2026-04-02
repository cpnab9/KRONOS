cmake_minimum_required(VERSION 3.10)
project(KRONOS_Trajectory_Optimizer)

set(CMAKE_CXX_STANDARD 17)

# 告诉 CMake 我们的自定义安装前缀
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# ==========================================
# 【新增】：告诉链接器 (ld) 去哪里找 libblasfeo.so 和 libfatrop.so
# ==========================================
link_directories("$ENV{HOME}/.local/lib")

# 查找 fatrop 包
find_package(fatrop REQUIRED)

if(fatrop_FOUND)
    message(STATUS "[KRONOS] Successfully found fatrop library!")
endif()

# ==========================================
# 编译离线生成的 C 代码
# ==========================================
add_library(casadi_generated SHARED generated/kronos_nlp_functions.c)
target_link_libraries(casadi_generated PRIVATE fatrop::fatrop)

# ==========================================
# 编译 KRONOS 在线主程序
# ==========================================
add_executable(kronos_mpc_node 
    src/main.cpp
    src/solver/fatrop_wrapper.cpp
)

target_include_directories(kronos_mpc_node PRIVATE 
    include
    generated
)

# 链接自动生成的底层库和 fatrop 求解器
target_link_libraries(kronos_mpc_node PRIVATE casadi_generated fatrop::fatrop)