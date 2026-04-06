```text
AeroPseudospectral-Planner/
├── CMakeLists.txt                  # 顶层 CMake 构建脚本
├── README.md                       # 工程说明文档
├── config/                         # 配置文件目录
│   └── planner_config.yaml         # 飞行器参数、最大网格数、约束边界等配置
├── scripts/                        # 离线模块：Python / CasADi 推导与代码生成
│   ├── models/                     # 飞行器动力学模型库 (如 固定翼、旋翼、火箭等)
│   ├── generator/                  # 伪谱法配置与矩阵生成核心逻辑
│   ├── generate_ocp.py             # 离线代码生成入口脚本
│   └── requirements.txt            # Python 依赖
├── include/                        # 在线模块：C++ 头文件 (接口与声明)
│   ├── memory/                     # 内存管理模块 (核心！实现零动态内存)
│   │   ├── static_pool.hpp         # 静态内存池定义
│   │   └── fixed_size_types.hpp    # 预定义的静态大小数组/结构体 (std::array)
│   ├── core/                       # 核心数据结构与枚举
│   │   └── trajectory_types.hpp    # 飞行器状态、控制指令的结构体定义
│   ├── ocp/                        # OCP (最优控制问题) 组装模块
│   │   ├── casadi_wrapper.hpp      # 包装 CasADi 生成的 C 纯函数接口
│   │   └── flight_ocp.hpp          # 继承 OcpAbstract，组装 Fatrop 问题
│   ├── planner/                    # 规划器顶层逻辑模块
│   │   ├── mesh_adapter.hpp        # 网格自适应逻辑 (计算需要的网格分布)
│   │   ├── warm_starter.hpp        # 热启动与轨迹移位预测逻辑
│   │   └── trajectory_planner.hpp  # 规划器主干，调度 OCP 和 Solver
│   └── utils/                      # 工具模块
│       └── timer.hpp               # 高精度耗时统计
├── src/                            # 在线模块：C++ 源文件 (实现)
│   ├── codegen/                    # 【自动生成目录】存放 CasADi 生成的 .c / .h 文件
│   │   └── casadi_codegen.c        # (由 scripts 生成，不提交到 git)
│   ├── ocp/
│   │   └── flight_ocp.cpp
│   ├── planner/
│   │   ├── mesh_adapter.cpp
│   │   ├── warm_starter.cpp
│   │   └── trajectory_planner.cpp
│   └── main.cpp                    # 独立的测试/运行入口
└── tests/                          # 单元测试 (Google Test)
    ├── test_memory_pool.cpp        # 内存泄露与静态分配测试
    ├── test_warm_start.cpp         # 验证移位热启动是否正确
    └── test_fatrop_solver.cpp      # 验证单次求解的正确性
```