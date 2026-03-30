cmake_minimum_required(VERSION 3.14)
project(KRONOS_Planner LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含目录
include_directories(include)

# ==============================================================================
# 1. 编译 CasADi 生成的 C 代码为一个静态库
# ==============================================================================
# 注意：不需要链接 CasADi 的 C++ 库，因为生成的是纯 C 代码，零依赖！
add_library(nlp_codegen STATIC
    generated/brachistochrone_nlp.c
)
# 开启 O3 优化，这对于底层的 Jacobian/Hessian 评估速度至关重要
target_compile_options(nlp_codegen PRIVATE -O3 -ffast-math)

# ==============================================================================
# 2. 编译测试程序
# ==============================================================================
add_executable(test_casadi_eval tests/test_casadi_eval.cpp)

# 链接目标：链接我们生成的 C 代码库，并引入 Eigen3
target_link_libraries(test_casadi_eval
    PRIVATE
    nlp_codegen
)

# ==============================================================================
# 3. 编译主在线规划程序
# ==============================================================================
add_executable(planner_main src/main.cpp)

target_link_libraries(planner_main
    PRIVATE
    nlp_codegen
    # Eigen3::Eigen (如果你用的是系统安装的方法一，就把这行取消注释；如果你用的是方法二，保持注释)
)