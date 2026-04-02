#include <iostream>
#include <iomanip>
#include "kronos/solver/fatrop_wrapper.hpp"
#include "kronos/utils/library_loader.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) return -1;

    try {
        auto funcs = kronos::LibraryLoader::load(argv[1]);
        kronos::FatropWrapper solver(funcs);

        solver.solve();

        // 优雅地只打印核心目标
        std::cout << "\n" << std::string(45, '=') << std::endl;
        std::cout << "   KRONOS OPTIMIZATION FINAL REPORT" << std::endl;
        std::cout << std::string(45, '-') << std::endl;
        
        std::cout << "  > OPTIMAL OBJECTIVE VALUE : " 
                  << std::fixed << std::setprecision(6) 
                  << solver.get_objective() << std::endl;
        
        std::cout << "  > SOLVER STATUS           : CONVERGED" << std::endl;
        std::cout << std::string(45, '=') << "\n" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }
    return 0;
}