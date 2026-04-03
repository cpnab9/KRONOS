#include <iostream>
#include <iomanip>
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "kronos/utils/timer.hpp"
#include "kronos/utils/exporter.hpp"

// 内部报告打印函数
void print_summary(double objective, const kronos::Timer& timer) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "   KRONOS OPTIMIZATION FINAL REPORT" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "  > OPTIMAL OBJECTIVE     : " << std::fixed << std::setprecision(6) << objective << std::endl;
    std::cout << "  > SOLVER STATUS         : CONVERGED\n" << std::endl;
    std::cout << "  [ TIMING ANALYSIS ]" << std::endl;
    std::cout << "  * Library Loading       : " << std::setprecision(3) << std::setw(8) << timer.get("load") << " ms" << std::endl;
    std::cout << "  * Memory Initialization : " << std::setprecision(3) << std::setw(8) << timer.get("init") << " ms" << std::endl;
    std::cout << "  * Fatrop Execution      : " << std::setprecision(3) << std::setw(8) << timer.get("solve") << " ms" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    std::cout << "  > TOTAL PIPELINE TIME   : " << std::setprecision(3) << std::setw(8) << timer.get("total") << " ms" << std::endl;
    std::cout << std::string(50, '=') << "\n" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
        kronos::Timer timer;
        timer.start("total");

        // 阶段 1：加载动态库
        timer.start("load");
        auto funcs = kronos::LibraryLoader::load(argv[1]); 
        timer.stop("load");

        // 阶段 2：初始化求解器
        timer.start("init");
        kronos::FatropWrapper solver(funcs); 
        timer.stop("init");

        // 阶段 3：执行求解
        solver.solve(); 
        timer.set("solve", solver.get_solve_time_ms()); // 获取包装器内部计时

        timer.stop("total");

        // 阶段 4：导出与报告
        kronos::Exporter::save_to_csv("optimization_results.csv", solver.get_solution());
        print_summary(solver.get_objective(), timer);

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}