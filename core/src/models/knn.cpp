#include "mlcore/models/knn.hpp"
#include <algorithm>
#include <queue>
#include <utility>

void kNNBase::check_size(std::size_t n) const {
    if (k_ > n)
        throw BadArgument("k exceeds training size");
}

std::vector<Neighbour> kNNBase::neighbours(std::span<const double> row) const {
    auto cmp = [](const Neighbour &a, const Neighbour &b) {
        return a.distance != b.distance ? a.distance < b.distance : a.index < b.index;
    };
    std::priority_queue<Neighbour, std::vector<Neighbour>, decltype(cmp)> heap(cmp);
    for (std::size_t i = 0; i < target_.size(); ++i) {
        Neighbour candidate{distance(row, i), i};
        if (heap.size() < k_)
            heap.push(candidate);
        else if (cmp(candidate, heap.top())) {
            heap.pop();
            heap.push(candidate);
        }
    }
    std::vector<Neighbour> out;
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    return out;
}

std::vector<double> kNNBase::weights(const std::vector<Neighbour> &ns) const {
    std::vector<double> out(ns.size(), 1);
    if (weighting_ == Weighting::Uniform)
        return out;
    double nearest = std::min_element(ns.begin(), ns.end(), [](auto a, auto b) {
                         return a.distance < b.distance;
                     })->distance;
    for (std::size_t i = 0; i < ns.size(); ++i)
        out[i] = nearest == 0 ? (ns[i].distance == 0 ? 1 : 0) : nearest / ns[i].distance;
    return out;
}

kNNBase::kNNBase(std::size_t k, std::unique_ptr<IDistance> d, Weighting w)
    : NeighbourModel(std::move(d)), k_(k), weighting_(w) {
    if (!k || (w != Weighting::Uniform && w != Weighting::Distance))
        throw BadArgument("Invalid kNN parameters");
}
