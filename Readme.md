KRONOS/
├── CMakeLists.txt                 # 顶层 CMake 构建配置
├── README.md                      # 项目说明文档
├── config/                        # 配置文件目录 (YAML/JSON)
│   ├── vehicle_params.yaml        # 飞行器气动、质量等物理参数
│   └── solver_settings.yaml       # fatrop 求解器参数 (最大迭代次数, 容差等)
│
├── offline_codegen/               # 【离线阶段】CasADi Python 脚本
│   ├── requirements.txt           # Python 依赖 (casadi, numpy等)
│   ├── models/                    # 连续动力学模型定义
│   │   ├── __init__.py
│   │   └── aircraft_6dof.py       # 例如：6自由度飞行器模型
│   ├── collocation/               # 伪谱法节点与权重计算工具
│   │   └── pseudospectral_utils.py 
│   └── generate_nlp_c_code.py     # 核心主脚本：导出 CasADi C 代码
│
├── generated/                     # 【自动生成】离线脚本生成的 C 代码存放地
│   ├── CMakeLists.txt             # 将生成的代码编译为静态库
│   ├── kronos_nlp_functions.c     # 包含目标、约束、Jacobian、Hessian的纯C代码
│   └── kronos_nlp_functions.h     # 对应的头文件
│
├── include/                       # 【在线阶段】C++ 头文件
│   └── kronos/
│       ├── core/
│       │   ├── kronos_types.hpp   # 基础数据类型与结构体定义
│       │   └── config_parser.hpp  # 解析 config 目录下的参数
│       ├── transcription/
│       │   ├── mesh_provider.hpp  # 提供伪谱法的时间网格与微分矩阵
│       │   └── pseudospectral_block.hpp # 处理单个伪谱段的变量映射
│       ├── solver/
│       │   └── fatrop_wrapper.hpp # 封装 fatrop 的调用逻辑与数据转换
│       └── guidance/
│           └── mpc_controller.hpp # 包含制导主循环、热启动逻辑的类
│
├── src/                           # 【在线阶段】C++ 源文件
│   ├── core/
│   │   └── config_parser.cpp
│   ├── transcription/
│   │   └── mesh_provider.cpp
│   ├── solver/
│   │   └── fatrop_wrapper.cpp     # 继承并实现 fatrop 接口，注入 generated 代码
│   ├── guidance/
│   │   └── mpc_controller.cpp
│   └── main.cpp                   # 机载端/仿真端程序入口点
│
├── third_party/                   # 第三方依赖 (作为 git submodule 引入)
│   └── fatrop/                    # 你提供的 fatrop 库放在这里
│
└── tests/                         # 单元测试与集成测试
    ├── CMakeLists.txt
    ├── test_mesh_generation.cpp
    └── test_solver_convergence.cpp