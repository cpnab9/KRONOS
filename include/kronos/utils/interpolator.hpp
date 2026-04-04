// KRONOS/include/kronos/utils/interpolator.hpp
#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace kronos {

class Interpolator1D {
public:
    // t: 时间节点序列, y: 对应时间点的状态/控制向量
    Interpolator1D(const std::vector<double>& t, const std::vector<std::vector<double>>& y)
        : t_(t), y_(y) {
        if (t.size() != y.size() || t.empty()) {
            throw std::invalid_argument("Interpolator1D: Time and Data dimensions mismatch or empty.");
        }
        dim_ = y[0].size();
    }

    // 传入目标时间，返回插值后的向量
    std::vector<double> operator()(double target_t) const {
        // 处理边界外推 (等同于 Python 的 fill_value)
        if (target_t <= t_.front()) return y_.front();
        if (target_t >= t_.back()) return y_.back();

        // 二分查找目标时间所在的区间
        auto it = std::lower_bound(t_.begin(), t_.end(), target_t);
        int idx = std::distance(t_.begin(), it) - 1;

        double t0 = t_[idx];
        double t1 = t_[idx + 1];
        double ratio = (target_t - t0) / (t1 - t0);

        std::vector<double> res(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            res[i] = y_[idx][i] + ratio * (y_[idx + 1][i] - y_[idx][i]);
        }
        return res;
    }

private:
    std::vector<double> t_;
    std::vector<std::vector<double>> y_;
    size_t dim_;
};

} // namespace kronos