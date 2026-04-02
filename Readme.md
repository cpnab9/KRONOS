# KRONOS: High-Performance Trajectory Optimizer

**KRONOS** is a trajectory optimization framework designed for high-performance motion planning and guidance. It leverages **Pseudospectral Collocation** methods and the **fatrop** solver to provide fast, numerically stable solutions for complex Optimal Control Problems (OCP).

## 🚀 Key Features

* **Pseudospectral Collocation**: Utilizes Radau points for high-accuracy discretization of continuous dynamics.
* **Offline Code Generation**: Uses CasADi in Python to pre-calculate Jacobians and Hessians, generating optimized C code for real-time execution.
* **Fast Solver Integration**: Built around the `fatrop` (Fast Trajectory Optimization) solver for high-speed convergence.
* **Modular C++ Architecture**: Provides a clean wrapper (`FatropWrapper`) to bridge generated NLP functions with the C++ runtime.

## 📂 Project Structure

```text
KRONOS/
├── offline_codegen/       # Python scripts for NLP definition and C-code generation
├── generated/             # Auto-generated C code (Target/Constraints/Jacobians)
├── include/               # C++ Headers (Solver wrappers, types, utilities)
├── src/                   # C++ Implementation (Fatrop integration, entry points)
├── config/                # Physical parameters and solver settings
└── third_party/           # External dependencies (fatrop)
```

## 🛠 Prerequisites

-   **Python 3.x** with `casadi` and `numpy`.
-   **C++ Compiler** (supporting C++17).
-   **CMake** (version 3.10 or higher).
-   **fatrop**: Must be installed or accessible via `CMAKE_PREFIX_PATH`.

## 🔄 Workflow

The KRONOS pipeline consists of two main stages:

### 1. Offline Phase (Code Generation)
Define your dynamics and constraints in Python. The script generates a pure C file containing the NLP functions required by the solver.

```bash
python3 offline_codegen/generate_brachistochrone.py
```
This produces `kronos_nlp_functions.c` and `.h` in the `generated/` directory.

### 2. Online Phase (Compilation & Execution)
CMake detects the generated code, compiles it into a shared library, and links it to the C++ runner.

```bash
mkdir build && cd build
cmake ..
make
```

## 🏃 Running Examples

### Brachistochrone Problem
A classic problem to find the path between two points that is covered by a point mass in the least time under constant gravity.

```bash
./run_brachistochrone
```
The program will output the optimal descent time (tf) and the terminal state reached.

## ⚙️ Configuration

* **Solver Settings**: Modify `fatrop` tolerances and iterations in `offline_codegen/` scripts.
* **Hardware Deployment**: The generated C code is self-contained (Zero-allocation mode), making it suitable for embedded flight computers.

--- 

### 💡 Note on Custom Problems
To implement a new trajectory optimization problem:
1. Create a Python script in `offline_codegen/` based on `generate_brachistochrone.py`.
2. Update the `CMakeLists.txt` to include the new target.
3. Implement a C++ runner in `src/` to call the `FatropWrapper`.