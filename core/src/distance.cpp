#include "mlcore/distance.hpp"
#include <cmath>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>

double EuclideanDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw std::runtime_error("different sizes");
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

double ManhattanDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw std::runtime_error("different sizes");
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

double ChebyshevDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw std::runtime_error("different sizes");
    double sum = 0;
    double res;
    for (std::size_t i = 0; i < a.size(); ++i) {
        res = std::abs(a[i] - b[i]);
        sum = res > sum ? res : sum;
    }
    return sum;
}

auto cmp = [](const Neighbour& a, const Neighbour& b) {
    if (a.distance != b.distance) {
        return a.distance < b.distance;
    }
    return a.index < b.index;
};

std::priority_queue<Neighbour, std::vector<Neighbour>, decltype(cmp)> heap(cmp);

kNNBase::kNNBase(std::size_t k, std::unique_ptr<IDistance> distance) : dist_(std::move(distance)), k_(k) { }



