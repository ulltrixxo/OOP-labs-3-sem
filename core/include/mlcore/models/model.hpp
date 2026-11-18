#pragma once
#include "mlcore/matrix.hpp"
class IModel {
  public:
    virtual ~IModel() = default;
    virtual void fit(const DatasetMatrix &, std::span<const double>) = 0;
    virtual std::vector<double> predict(const DatasetMatrix &) const = 0;
};
class IRegressor : public virtual IModel {};
class IClassifier : public virtual IModel {
  public:
    virtual DatasetMatrix predict_proba(const DatasetMatrix &) const = 0;
    virtual std::span<const double> classes() const = 0;
};
