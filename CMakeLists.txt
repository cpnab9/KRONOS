cmake_minimum_required(VERSION 3.14)
project(AeroPseudospectralPlanner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 1. 查找底层的 Fatrop 求解器库
find_package(fatrop REQUIRED)

# 2. 自动拉取并配置 nlohmann/json 库 (无需手动下载安装)
include(FetchContent)
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# 3. 添加头文件搜索路径
include_directories(include)

# 4. 收集所有源文件 (包含统一的生成代码 casadi_codegen.c)
set(SRC_FILES
    src/codegen/casadi_codegen.c
    src/ocp/flight_ocp.cpp
    src/planner/trajectory_planner.cpp
)

# 5. 编译可执行文件
add_executable(planner_node 
    ${SRC_FILES}
    src/main.cpp
)

# 6. 链接所有依赖库 (Fatrop 和 JSON库)
target_link_libraries(planner_node 
    PRIVATE 
    fatrop::fatrop
    nlohmann_json::nlohmann_json
)