#pragma once
#include "mlcore/models/neighbour_model.hpp"
class kNNBase : public NeighbourModel {
    std::size_t k_;
    Weighting weighting_;

  protected:
    void check_size(std::size_t n) const override;
    std::vector<Neighbour> neighbours(std::span<const double> row) const override;
    std::vector<double> weights(const std::vector<Neighbour> &ns) const override;

  public:
    kNNBase(std::size_t k, std::unique_ptr<IDistance> d, Weighting w = Weighting::Uniform);
};
class kNNRegressor : public kNNBase, public IRegressor {
  public:
    using kNNBase::kNNBase;
};
using kNNClassifier = ClassifierModel<kNNBase>;
using KnnRegressor = kNNRegressor;
using KnnClassifier = kNNClassifier;
