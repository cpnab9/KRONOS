# KRONOS: High-Performance Online Flight Trajectory Optimization

**KRONOS** is a high-performance C++ framework designed for real-time trajectory optimization of aerospace vehicles, with a specific focus on hypersonic flight dynamics. It leverages state-of-the-art numerical optimization techniques to solve complex Non-Linear Programming (NLP) problems under stringent physical constraints and limited computational budgets.

## 🚀 Key Features

* **Structure-Aware Optimization**: Built on top of the **Fatrop** solver, KRONOS exploits the $O(N)$ linear complexity of Optimal Control Problems (OCP) using structured Riccati recursion.
* **Automatic Differentiation**: Integrated with **CasADi** for high-efficiency symbolic differentiation and C-code generation of dynamics, Jacobians, and Hessians.
* **Native Slack Variables**: Robust handling of path constraints (e.g., heat flux, dynamic pressure, load factor) through native soft-constraint support, preventing solver infeasibility in extreme flight envelopes.
* **Intelligent Initialization**: Implements high-quality initial guess generation via linear interpolation between user-defined endpoints, significantly reducing cold-start iteration counts.
* **Onboard-Ready Architecture**: Decoupled Python-based modeling and C++ execution engine, optimized for deployment on resource-constrained embedded systems (e.g., ARM, Jetson).

## 🛠 System Architecture

KRONOS follows a two-stage workflow:
1.  **Offline/Modeling (Python)**: Define vehicle dynamics, Radau pseudospectral discretization, and constraints using CasADi. Export high-performance C-code and JSON-based problem configurations.
2.  **Online/Execution (C++)**: The `TrajectoryPlanner` engine loads the problem configuration and interfaces with the generated code to solve the OCP in real-time.

## 📦 Prerequisites

* **Fatrop**: Structured NLP solver for OCP.
* **CasADi**: For symbolic modeling and code generation.
* **BLASFEO**: High-performance linear algebra kernels.
* **nlohmann_json**: For parsing OCP configurations.

## 🚀 Quick Start

### 1. Generate the Model (Python)
Define your flight mission in the generator script and run it to produce the C kernels:
```bash
python scripts/generate_flight.py
```

### 2. Build the C++ Engine
```bash
mkdir build && cd build
cmake ..
make -j
```

### 3. Run the Planner
```bash
./KRONOS_main
```

## 📈 Performance Benchmarks

* **Cold Start**: ~500ms for a complex hypersonic reentry problem with 50 nodes and 5th-order Radau collocation (145 iterations).
* **Warm Start**: Expected <50ms for rolling-horizon MPC applications, enabling 20Hz+ onboard planning rates.

## 🗺 Roadmap

- [ ] **Adaptive Mesh Refinement**: Dynamic grid allocation to balance accuracy and compute time.
- [ ] **Hardware-in-the-Loop (HIL)**: Validation on ARM-based flight control computers.
- [ ] **Robust NMPC**: Online obstacle/no-fly zone avoidance under uncertainty.

## 📄 License
This project is licensed under the MIT License.

---

### Core Classes reference:
* **`FlightOCP`**: Implementation of the `fatrop::OcpAbstract` interface, handling matrix assembly for dynamics and slack variables.
* **`TrajectoryPlanner`**: High-level manager for configuration loading and solver execution.
* **`OCPConfig`**: Data structure holding problem dimensions, bounds, and initial guesses.