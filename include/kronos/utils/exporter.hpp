#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include "problem_metadata.h"

namespace kronos {

class Exporter {
public:
    static void save_to_csv(const std::string& filename, const std::vector<double>& solution) {
        std::ofstream file(filename);
        if (!file.is_open()) return;

        // 写入表头 (使用元数据宏)
        file << "time," << KRONOS_X_NAMES << "," << KRONOS_U_NAMES << "\n";

        size_t ptr = 0;
        double tf = solution[KRONOS_TF_INDEX]; 
        double dt = tf / KRONOS_N; 
        const std::vector<double> tau = KRONOS_TAU_ROOT;

        // 1. 输出起点 (t=0)
        file << 0.0;
        for (int i = 0; i < KRONOS_NX; ++i) file << "," << solution[ptr++];
        for (int i = 0; i < KRONOS_NU_BASE; ++i) file << ",0.0";
        file << "\n";

        // 2. 遍历 N 个区间解包配点
        for (int k = 0; k < KRONOS_N; ++k) {
            size_t x_c_ptr = ptr;
            ptr += KRONOS_D * KRONOS_NX;
            size_t u_c_ptr = ptr;
            ptr += KRONOS_D * KRONOS_NU_BASE;

            double t_start = k * dt;
            for (int j = 0; j < KRONOS_D; ++j) {
                file << (t_start + tau[j] * dt);
                for (int i = 0; i < KRONOS_NX; ++i) 
                    file << "," << solution[x_c_ptr + j * KRONOS_NX + i];
                for (int i = 0; i < KRONOS_NU_BASE; ++i) 
                    file << "," << solution[u_c_ptr + j * KRONOS_NU_BASE + i];
                file << "\n";
            }
            ptr += KRONOS_NX; // 跨过主节点 X_{k+1}
        }
        file.close();
        std::cout << "  > RESULTS EXPORTED TO   : " << filename << std::endl;
    }
};

} // namespace kronos