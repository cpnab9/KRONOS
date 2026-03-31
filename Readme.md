# KRONOS: KKT Real-time Onboard Newton Optimization Solver

**KRONOS** (named after the Greek god of time) is a high-performance C++ non-linear programming (NLP) solver specifically designed for embedded and real-time systems. It focuses on solving complex optimization problems with both equality and inequality constraints using a **Primal-Dual Interior Point Method (IPM)**.

---

## 🚀 Key Features

* **Full Interior Point Method**: Supports non-linear equality constraints $g(w)=0$ and inequality constraints $h(w) \ge 0$.
* **Matrix Condensation**: Uses algebraic elimination to compress inequality constraints into the Hessian matrix, keeping the core KKT system dimension constant for maximum efficiency.
* **Schur Complement Solver**: Features a high-performance linear system solver based on the Schur Complement.
* **CasADi Integration**: Leverages CasADi's symbolic math and Automatic Differentiation (AD) to generate pre-compiled C code, eliminating runtime overhead for Jacobian and Hessian computations.
* **Hybrid Line Search Strategy**: Combines an $L_1$ exact merit function with KKT residual descent criteria to ensure robust convergence for highly non-linear problems like the Brachistochrone.

---

## 🛠️ Project Structure

```text
KRONOS/
├── scripts/           # Python scripts: Define problems using CasADi and generate C code
├── generated/         # Auto-generated C code and configs (kkt_funcs.c, kronos_config.h)
├── include/           # Core algorithm headers
│   ├── kronos_types.hpp       # Type definitions (Eigen wrappers)
│   ├── kronos_nlp_wrapper.hpp # NLP interface wrapper
│   ├── kronos_kkt_solver.hpp  # KKT linear system solver
│   └── kronos_optimizer.hpp   # IPM optimizer logic
├── src/               # Core algorithm implementation
│   ├── kronos_nlp_wrapper.cpp
│   ├── kronos_kkt_solver.cpp
│   ├── kronos_optimizer.cpp
│   └── main.cpp               # Entry point
└── third_party/       # Third-party libraries (primarily Eigen)
```

---

## 📐 Mathematical Background

KRONOS solves standard NLP problems of the form:

$$\min_w f(w)$$
$$\text{s.t.} \quad g(w) = 0, \quad h(w) \ge 0$$

To handle inequalities, we introduce **barrier functions** and **slack variables** $s$:

$$\min_{w, s} f(w) - \mu \sum \ln(s_i)$$
$$\text{s.t.} \quad g(w) = 0, \quad h(w) - s = 0$$

By utilizing **condensation** techniques in the KKT system, we process inequality constraints in real-time without increasing the primary system's dimensionality.

---

## 💻 Quick Start

### 1. Generate Problem Code
Define your optimization problem in a Python script. For example, to generate code for the Brachistochrone problem:

```bash
# Run the script to generate C interfaces
python3 scripts/codegen_brachistochrone.py
```

### 2. Build the Project
The project uses CMake for building. Ensure you have a C++ compiler configured.

```bash
mkdir build && cd build
cmake ..
make
```

### 3. Run the Solver
Execute the binary to observe the optimization iteration process:

```bash
./kronos
```

---

## 📊 Performance

KRONOS utilizes a **Primal-Dual Step Splitting** strategy to effectively handle non-convex trajectory optimization problems.

* **Convergence Criteria**: Uses the infinity norm of KKT conditions $\| \text{Res} \|_\infty < 10^{-5}$.
* **Line Search**: Integrated backtracking method with Armijo conditions to automatically adapt to non-linear curvature.

---

## 📚 Dependencies

* **C++17** or higher.
* **Eigen 3**: For matrix operations (included in `third_party`).
* **CasADi (Python)**: Required only for the offline code generation phase.

---
*"Guided by the God of Time, optimizing every millisecond."*