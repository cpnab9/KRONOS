cmake_minimum_required(VERSION 3.10)
project(KRONOS_Trajectory_Optimizer)

set(CMAKE_CXX_STANDARD 17)

# 指定 WSL/Linux 下的自定义安装前缀
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")
link_directories("$ENV{HOME}/.local/lib")

# 查找 fatrop 包
find_package(fatrop REQUIRED)

if(fatrop_FOUND)
    message(STATUS "[KRONOS] Successfully found fatrop library!")
endif()

# ==========================================
# 算例 1：最速降线问题 (Brachistochrone)
# ==========================================
# 检查底层的 C 文件是否存在，存在才执行编译
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/kronos_nlp_functions.c")
    # 编译专属的动态库
    add_library(brachio_generated SHARED generated/kronos_nlp_functions.c)
    target_link_libraries(brachio_generated PRIVATE fatrop::fatrop)

    # 编译专属的可执行文件
    add_executable(run_brachistochrone 
        src/run_brachistochrone.cpp
        src/solver/fatrop_wrapper.cpp
    )
    target_include_directories(run_brachistochrone PRIVATE include generated)
    target_link_libraries(run_brachistochrone PRIVATE brachio_generated fatrop::fatrop)
    
    message(STATUS "[KRONOS] Configured target: run_brachistochrone")
endif()

# ==========================================
# 算例 2：带约束的轨迹优化问题 (Constrained Ascent)
# ==========================================
# 目前这个文件还没生成，等我们下一步写好 Python 脚本后，CMake 会自动识别它
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/generated/constrained_nlp_functions.c")
    add_library(constrained_generated SHARED generated/constrained_nlp_functions.c)
    target_link_libraries(constrained_generated PRIVATE fatrop::fatrop)

    add_executable(run_simple_constrained 
        src/run_simple_constrained.cpp
        src/solver/fatrop_wrapper.cpp
    )
    target_include_directories(run_simple_constrained PRIVATE include generated)
    target_link_libraries(run_simple_constrained PRIVATE constrained_generated fatrop::fatrop)
    
    message(STATUS "[KRONOS] Configured target: run_simple_constrained")
endif()c