cmake_minimum_required(VERSION 3.10)
project(AeroPseudospectralPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 Fatrop
find_package(fatrop REQUIRED)

# 添加头文件搜索路径
include_directories(include)

# 收集所有源文件 (排除掉 main，便于未来编译为动态库)
set(SRC_FILES
    src/codegen/casadi_codegen.c
    src/ocp/flight_ocp.cpp
    src/planner/trajectory_planner.cpp
)

# 编译可执行文件
add_executable(planner_node 
    ${SRC_FILES}
    src/main.cpp
)

# 链接库
target_link_libraries(planner_node PRIVATE fatrop::fatrop)