#include "mlcore/models/knn.hpp"
#include "mlcore/exceptions.hpp"
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <span>
#include <utility>
#include <vector>

auto cmp = [](const Neighbour& a, const Neighbour& b) {
    if (a.distance != b.distance) {
        return a.distance < b.distance;
    }
    return a.index < b.index;
};

kNNBase::kNNBase(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting) : dist_(std::move(distance)),
 k_(k), weighting_(weighting), train_data_(), target_() {
    if (!dist_) throw BadArgument(diagnostics::message("Distance metric must not be null"));
    if (k_ == 0) throw BadArgument(diagnostics::message("k must be at least 1"));
    if (weighting_ != Weighting::Uniform && weighting_ != Weighting::Distance)
        throw BadArgument(diagnostics::message("Unknown weighting policy"));
}

kNNRegressor::kNNRegressor(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting) : kNNBase(k, std::move(distance), weighting) {}

kNNClassifier::kNNClassifier(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting) : kNNBase(k, std::move(distance), weighting) {}


void kNNBase::fit(const DatasetMatrix& features, std::span<const double> target) {
    if (features.get_rows() == 0 || features.get_cols() == 0)
        throw BadArgument(diagnostics::message("Training data must have rows and features"));
    if (k_ > features.get_rows())
        throw BadArgument(diagnostics::message("k exceeds the training row count"));
    if (target.size() != features.get_rows())
        throw SizeError(diagnostics::message("Target count must equal training row count"));
    if (features.get_layout() != Layout::RowMajor)
        throw MatrixTypeError(diagnostics::message("kNN requires RowMajor storage"));

    train_data_ = features;
    target_ = std::vector<double>(target.begin(), target.end());
    fitted_ = true;
}

std::vector<double> kNNBase::predict(const DatasetMatrix& features) const {
    if (!fitted_) throw ModelFitError(diagnostics::message(
        "Cannot call predict: the model is not fitted; call fit first"));
    if (features.get_cols() != train_data_.get_cols())
        throw SizeError(diagnostics::message("Query feature count differs from training"));
    if (features.get_layout() != Layout::RowMajor)
        throw MatrixTypeError(diagnostics::message("kNN requires RowMajor storage"));

    std::vector<double> result;
    result.resize(features.get_rows());
    double distance = 0;
    for (std::size_t row = 0; row < features.get_rows(); ++row) {
        std::priority_queue<Neighbour, std::vector<Neighbour>, decltype(cmp)> heap(cmp);
        for (std::size_t i = 0; i < train_data_.get_rows(); ++i) {
            distance = dist_->compute(features[row], train_data_[i]);
            if (distance < 0)
                throw BadArgument(diagnostics::message("Metric returned negative distance"));
            if (heap.size() < k_) {
                heap.push({distance, i});
            } else if (cmp({distance, i}, heap.top())) {
                heap.pop(); heap.push({distance, i});
            }
        }
        std::vector<Neighbour> neighbours;
        neighbours.reserve(heap.size());
        while (!heap.empty()) {
            neighbours.push_back(heap.top());
            heap.pop();
        }
        result[row] = consolidate(neighbours);
    }
    return result;
}

std::vector<double> kNNBase::weights(const std::vector<Neighbour>& neighbours) const {
    std::vector<double> result(neighbours.size(), 1.0); // вектор весов

    if (weighting_ == Weighting::Uniform) return result; // все веса 1

    double nearest = std::numeric_limits<double>::infinity();
    for (const auto& neighbour : neighbours)
        nearest = std::min(nearest, neighbour.distance); // ищем минимальное расстояние
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
        // даём вес с условием нулевого расстояния
        result[i] = nearest == 0 ? (neighbours[i].distance == 0 ? 1.0 : 0.0) : nearest / neighbours[i].distance;
    }
    return result;
}

double kNNRegressor::consolidate(const std::vector<Neighbour>& neighbours) const {
    const std::vector<double> w = weights(neighbours); // вектор весов
    double sum = 0, total = 0;
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
        sum += w[i] * target_[neighbours[i].index];
        total += w[i];
    }
    return sum / total;
}

double kNNClassifier::consolidate(const std::vector<Neighbour>& neighbours) const {
    const std::vector<double> w = weights(neighbours); // вектор весов
    std::map<double, double> votes; // голоса
    for (std::size_t i = 0; i < neighbours.size(); ++i)
        votes[target_[neighbours[i].index]] += w[i]; // прибавляем вес
    double result = 0;
    double max_vote = -1;
    for (const auto& [label, vote] : votes) {
        if (vote > max_vote) {
            max_vote = vote;
            result = label;
        }
    }
    return result;
}
