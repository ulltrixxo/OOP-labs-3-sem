#include "mlcore/models/neighbour_model.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

NeighbourModel::NeighbourModel(std::unique_ptr<IDistance> d) : dist_(std::move(d)) {
    if (!dist_)
        throw BadArgument("Null metric");
}

// проверить признаки на конечность
void check_features(const DatasetMatrix &x) {
    for (std::size_t i = 0; i < x.get_rows(); ++i)
        for (std::size_t j = 0; j < x.get_cols(); ++j)
            if (!std::isfinite(x(i, j)))
                throw BadArgument("Features must be finite");
}

//проверить обучающую и тестовую выборку
void check_training(const DatasetMatrix &x, std::span<const double> y) {
    if (!x.get_rows() || !x.get_cols())
        throw BadArgument("Empty training set");
    if (x.get_rows() != y.size())
        throw SizeError("Target count differs from row count");
    check_features(x);
    for (double v : y)
        if (!std::isfinite(v))
            throw BadArgument("Targets must be finite");
}

// создание копии RowMajor
DatasetMatrix dense_copy(const DatasetMatrix &x) {
    DatasetMatrix out(x.get_rows(), x.get_cols(), Layout::RowMajor, x.get_feature_names());
    for (std::size_t i = 0; i < x.get_rows(); ++i)
        for (std::size_t j = 0; j < x.get_cols(); ++j)
            out(i, j) = x(i, j);
    return out;
}

// вернуть различные названия категориальных признаков
std::vector<double> class_labels(std::span<const double> y) {
    std::vector<double> c(y.begin(), y.end());
    std::sort(c.begin(), c.end());
    c.erase(std::unique(c.begin(), c.end()), c.end());
    return c;
}


void NeighbourModel::check_size(std::size_t) const {}

// вычислить среднее среди расстояний с учетом весов
double NeighbourModel::consolidate(const std::vector<Neighbour> &ns) const {
    std::vector<double> w = weights(ns);
    long double sum = 0, total = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        sum += static_cast<long double>(w[i]) * target_[ns[i].index];
        total += w[i];
    }
    return static_cast<double>(sum / total);
}

void NeighbourModel::ready() const { if (!fitted_) throw ModelFitError("Call fit first"); }

// проверка набора объектов
void NeighbourModel::check_query(const DatasetMatrix &x) const {
    ready();
    if (x.get_cols() != train_data_.get_cols())
        throw SizeError("Wrong feature count");
    check_features(x);
}

// вычислить расстояние
double NeighbourModel::distance(std::span<const double> a, std::size_t i) const {
    double d = dist_->compute(a, train_data_[i]);
    if (!std::isfinite(d) || d < 0)
        throw BadArgument("Distance outside supported numerical range");
    return d;
}


// ввести обучающую выборку
void NeighbourModel::fit(const DatasetMatrix &x, std::span<const double> y) {
    fitted_ = false;
    check_training(x, y);
    check_size(x.get_rows());
    train_data_ = dense_copy(x);
    target_.assign(y.begin(), y.end());
    classes_ = class_labels(y);
    fitted_ = true;
}

// предсказать
std::vector<double> NeighbourModel::predict(const DatasetMatrix &x) const {
    check_query(x);
    std::vector<double> result(x.get_rows());

    std::vector<double> row(x.get_cols());
    for (std::size_t i = 0; i < x.get_rows(); ++i) {
        for (std::size_t j = 0; j < x.get_cols(); ++j)
            row[j] = x(i, j);
        result[i] = consolidate(neighbours(row));
    }
    return result;
}
