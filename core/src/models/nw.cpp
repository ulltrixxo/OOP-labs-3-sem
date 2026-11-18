#include "mlcore/models/nw.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

// найти все расстояния
std::vector<Neighbour> NwBase::neighbours(std::span<const double> row) const {
    std::vector<Neighbour> ns;
    ns.reserve(target_.size());
    for (std::size_t i = 0; i < target_.size(); ++i)
        ns.push_back({distance(row, i), i});
    return ns;
}

// вычислить веса
std::vector<double> NwBase::weights(const std::vector<Neighbour> &ns) const {
    std::vector<long double> logs;
    logs.reserve(ns.size());
    for (auto n : ns) {
        long double ratio = static_cast<long double>(n.distance) / bandwidth_;
        long double a = -0.5L * ratio * ratio; // показатель степени
        if (!std::isfinite(a))
            throw BadArgument("Gaussian log weight overflow");
        logs.push_back(a);
    }
    long double maximum = *std::max_element(logs.begin(), logs.end()); // макс элемент (наим расстояние)
    std::vector<double> out(logs.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = static_cast<double>(std::exp(logs[i] - maximum));
    return out;
}

NwBase::NwBase(double h, std::unique_ptr<IDistance> d)
    : NeighbourModel(std::move(d)), bandwidth_(h) {
    if (!std::isfinite(h) || h <= 0)
        throw BadArgument("Bandwidth must be finite and positive");
}