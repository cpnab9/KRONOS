KRONOS (KKT Real-time Onboard Newton Optimization Solver)
柯罗诺斯（希腊神话中的时间之神）

KRONOS/
├── CMakeLists.txt
├── generated/ (kkt_funcs.c, kkt_funcs.h, kronos_config.h)
├── include/
│   ├── kronos_types.hpp
│   ├── kronos_nlp_wrapper.hpp
│   ├── kronos_kkt_solver.hpp
│   └── kronos_optimizer.hpp
├── src/
|   ├── kronos_nlp_wrapper.cpp
|   ├── kronos_kkt_solver.cpp
|   ├── kronos_optimizer.cpp
|   └── main.cpp
└── third_party/ (Eigen)