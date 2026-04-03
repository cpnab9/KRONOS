#pragma once
#include <chrono>
#include <string>
#include <map>
#include <vector>

namespace kronos {

class Timer {
public:
    // 开始记录某个阶段
    void start(const std::string& name) {
        starts_[name] = std::chrono::high_resolution_clock::now();
    }

    // 停止记录某个阶段
    void stop(const std::string& name) {
        auto end = std::chrono::high_resolution_clock::now();
        if (starts_.count(name)) {
            durations_[name] = std::chrono::duration<double, std::milli>(end - starts_[name]).count();
        }
    }

    // 直接设置耗时（例如从求解器获取的内部时间）
    void set(const std::string& name, double ms) {
        durations_[name] = ms;
    }

    double get(const std::string& name) const {
        return durations_.at(name);
    }

private:
    std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> starts_;
    std::map<std::string, double> durations_;
};

} // namespace kronos