#pragma once
#include "mlcore/models/neighbour_model.hpp"
class NwBase : public NeighbourModel {
    double bandwidth_; // h

  protected:
    std::vector<Neighbour> neighbours(std::span<const double> row) const override;
    std::vector<double> weights(const std::vector<Neighbour> &ns) const override;

  public:
    NwBase(double h, std::unique_ptr<IDistance> d);
};
class NwRegressor : public NwBase, public IRegressor {
  public:
    using NwBase::NwBase;
};
using NwClassifier = ClassifierModel<NwBase>;
