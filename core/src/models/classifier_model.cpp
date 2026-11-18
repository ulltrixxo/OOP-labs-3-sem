#include "mlcore/models/knn.hpp"
#include "mlcore/models/nw.hpp"
#include <algorithm>

template <class Base>
std::vector<double> ClassifierModel<Base>::probabilities(const std::vector<Neighbour> &ns) const {
    auto w = this->weights(ns);
    std::vector<long double> votes(this->classes_.size(), 0);
    long double total = 0;
    for (std::size_t i = 0; i < ns.size(); ++i) {
        auto c = std::lower_bound(this->classes_.begin(), this->classes_.end(),
                                  this->target_[ns[i].index]) -
                 this->classes_.begin();
        votes[c] += w[i];
        total += w[i];
    }
    std::vector<double> out(votes.size());
    for (std::size_t c = 0; c < out.size(); ++c)
        out[c] = static_cast<double>(votes[c] / total);
    return out;
}

template <class Base>
double ClassifierModel<Base>::consolidate(const std::vector<Neighbour> &ns) const {
    auto p = probabilities(ns);
    return this->classes_[std::max_element(p.begin(), p.end()) - p.begin()];
}

template <class Base>
std::span<const double> ClassifierModel<Base>::classes() const {
    this->ready();
    return this->classes_;
}

template <class Base>
DatasetMatrix ClassifierModel<Base>::predict_proba(const DatasetMatrix &features) const {
    this->check_query(features);

    DatasetMatrix out(features.get_rows(), this->classes_.size());
    std::vector<double> row(features.get_cols());
    for (std::size_t i = 0; i < features.get_rows(); ++i) {
        for (std::size_t j = 0; j < features.get_cols(); ++j)
            row[j] = features(i, j);
        std::vector<double> p = probabilities(this->neighbours(row));
        for (std::size_t c = 0; c < p.size(); ++c)
            out(i, c) = p[c];
    }
    return out;
}

// Определения шаблона скрыты в cpp. Создаём код для двух используемых баз.
// При добавлении классификатора с другой Base нужно добавить инстанцирование здесь.
template class ClassifierModel<kNNBase>;
template class ClassifierModel<NwBase>;
