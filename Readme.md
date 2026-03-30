KRONOS (KKT Real-time Onboard Newton Optimization Solver)
柯罗诺斯（希腊神话中的时间之神）

trajectory_planner/
├── CMakeLists.txt              # 项目的根 CMake 构建脚本
├── README.md                   # 项目说明文档
├── build/                      # 编译生成目录 (执行 cmake .. 和 make 的地方)
├── config/                     # 配置文件目录 (存放飞行器参数、求解器参数等)
│   └── planner_config.json     
├── scripts/                    # 离线处理脚本 (Python)
│   ├── vehicle_dynamics.py     # 飞行器动力学模型的符号化定义
│   ├── pseudospectral.py       # 伪谱法离散化逻辑 (LGL/CGL节点计算等)
│   └── generate_nlp_casadi.py  # 核心脚本：组装NLP并调用 CasADi 生成 .c 文件
├── generated/                  # CasADi 自动生成的 C 代码存放处 (由 scripts 导出)
│   └── brachistochrone_nlp.c   # 包含段内(Local)和段间(Link)所有的 Jacobian, Hessian 等评估函数
├── include/                    # C++ 头文件 (.hpp/.h)
│   ├── dynamics/
│   │   └── vehicle_model.hpp   # C++ 端的飞行器参数与接口定义
│   ├── pseudospectral/
│   │   └── mesh_manager.hpp    # 网格管理、分段区间更新、节点分配
│   ├── solver/
│   │   ├── ipm_core.hpp        # 内点法主流程 (线搜索、阻尼更新等)
│   │   ├── kkt_system.hpp      # KKT 矩阵组装
│   │   └── schur_complement.hpp# 舒尔补求解器 (专门处理 KKT 的块箭头稀疏结构)
│   └── utils/
│       ├── casadi_wrapper.hpp  # 封装 generated 目录下生成的 C 函数
│       └── timer.hpp           # 性能测试工具 (记录 Jacobian/Hessian 评估耗时)
├── src/                        # C++ 源文件 (.cpp)
│   ├── main.cpp                # 在线规划入口程序
│   ├── dynamics/
│   │   └── vehicle_model.cpp   
│   ├── pseudospectral/
│   │   └── mesh_manager.cpp    
│   ├── solver/
│   │   ├── ipm_core.cpp        
│   │   ├── kkt_system.cpp      
│   │   └── schur_complement.cpp
│   └── utils/
│       └── casadi_wrapper.cpp  
└── tests/                      # 单元测试代码
    ├── test_casadi_eval.cpp    # 测试导数生成是否正确
    └── test_schur_solver.cpp   # 测试舒尔补线性方程组求解精度