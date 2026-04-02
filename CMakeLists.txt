cmake_minimum_required(VERSION 3.10)
project(KRONOS_Trajectory_Optimizer)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 环境与依赖配置 (WSL/Linux)
# ==========================================
# 指定 fatrop 和 blasfeo 的安装路径
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
link_directories("$ENV{HOME}/.local/lib")

# 查找 fatrop 包
find_package(fatrop REQUIRED)

# ==========================================
# 2. 编译生成的算例库 (Plugin)
# ==========================================
# 严格按照你 Python 脚本导出的文件名：generated/nlp_code.c
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/nlp_code.c")
    # 将生成的 C 代码编译为共享库 nlp_case.so
    add_library(nlp_case SHARED generated/nlp_code.c)
    target_link_libraries(nlp_case PRIVATE fatrop::fatrop)
    
    # 去掉 lib 前缀，直接生成 nlp_case.so 方便 C++ 加载
    set_target_properties(nlp_case PROPERTIES PREFIX "")
    message(STATUS "[KRONOS] Found nlp_code.c, target nlp_case configured.")
else()
    message(WARNING "[KRONOS] generated/nlp_code.c not found. Please run Python script first.")
endif()

# ==========================================
# 3. 编译通用求解器程序 (Solver Engine)
# ==========================================
add_executable(kronos_solver 
    src/main.cpp
    src/solver/fatrop_wrapper.cpp
)

# 包含头文件
target_include_directories(kronos_solver PRIVATE include)

# 链接依赖：求解器核心 + Linux 动态加载库 (dl)
target_link_libraries(kronos_solver PRIVATE 
    fatrop::fatrop 
    ${CMAKE_DL_LIBS}
)