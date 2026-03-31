KRONOS (KKT Real-time Onboard Newton Optimization Solver)
柯罗诺斯（希腊神话中的时间之神）

KRONOS/
├── CMakeLists.txt
├── generated/ (kkt_funcs.c, kkt_funcs.h, kronos_config.h)
├── include/
│   ├── kronos_types.hpp          (基础类型定义)
│   ├── kronos_nlp_wrapper.hpp    (NLP接口层)
│   └── kronos_solver.hpp         (牛顿-舒尔补求解器)
├── src/
│   ├── kronos_nlp_wrapper.cpp
│   ├── kronos_solver.cpp
│   └── main.cpp                  (极简主程序)
└── third_party/ (Eigen)