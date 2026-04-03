#include <iostream>
#include <iomanip>
#include <vector>
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"
#include "kronos/utils/timer.hpp"
#include "kronos/utils/exporter.hpp"
#include "kronos/utils/mesh_refiner.hpp"

void print_summary(int iter, double objective, double solve_time_ms) {
    std::cout << "  > [Iter " << iter << "] OPTIMAL OBJECTIVE : " << std::fixed << std::setprecision(6) << objective << std::endl;
    std::cout << "  > [Iter " << iter << "] FATROP EXEC TIME  : " << std::setprecision(3) << solve_time_ms << " ms" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_shared_library>" << std::endl;
        return -1;
    }

    try {
        kronos::Timer timer;
        timer.start("total");

        auto funcs = kronos::LibraryLoader::load(argv[1]); 
        kronos::FatropWrapper solver(funcs); 

        const int max_adapt_iters = 3;
        int N = kronos::MeshRefiner::N;
        
        std::vector<double> current_fractions(N, 1.0 / N);
        
        // 【关键修复】：第一轮直接使用构建好的物理插值！
        std::vector<double> current_x0 = kronos::MeshRefiner::build_initial_guess();

        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "   KRONOS ADAPTIVE MESH OPTIMIZATION PIPELINE" << std::endl;
        std::cout << std::string(50, '=') << std::endl;

        for (int iter = 1; iter <= max_adapt_iters; ++iter) {
            std::cout << "[*] Running Iteration " << iter << " / " << max_adapt_iters << " ..." << std::endl;

            // 每次迭代前，均注入参数与合法初值
            solver.set_mesh_fractions(current_fractions);
            solver.set_initial_guess(current_x0);

            solver.solve(); 
            
            print_summary(iter, solver.get_objective(), solver.get_solve_time_ms());

            if (iter == max_adapt_iters) break;

            // 读取成功解，生成更精准的网格参数与 Warm-start 初值
            const auto& sol = solver.get_solution();
            auto refine_res = kronos::MeshRefiner::compute(current_fractions, sol);
            
            current_fractions = refine_res.new_fractions;
            current_x0 = refine_res.new_initial_guess;
        }

        timer.stop("total");

        kronos::Exporter::save_to_csv("adaptive_optimization_results.csv", solver.get_solution());
        std::cout << "  > TOTAL PIPELINE TIME   : " << std::setprecision(3) << timer.get("total") << " ms\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}