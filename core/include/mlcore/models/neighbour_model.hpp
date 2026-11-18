#pragma once
#include "mlcore/distance.hpp"
#include "mlcore/exceptions.hpp"
#include "mlcore/models/model.hpp"
#include <memory>
#include <vector>
#include <span>
#include <cstddef>

void check_features(const DatasetMatrix &x);
void check_training(const DatasetMatrix &x, std::span<const double> y);
DatasetMatrix dense_copy(const DatasetMatrix &x);
std::vector<double> class_labels(std::span<const double> y);
struct Neighbour {
    double distance;
    std::size_t index;
};
enum class Weighting { Uniform, Distance };

class NeighbourModel : public virtual IModel {
  protected:
    std::unique_ptr<IDistance> dist_;
    DatasetMatrix train_data_;
    std::vector<double> target_, classes_;
    bool fitted_ = false;

    virtual void check_size(std::size_t) const;
    virtual std::vector<Neighbour> neighbours(std::span<const double>) const = 0;
    virtual std::vector<double> weights(const std::vector<Neighbour> &) const = 0;
    virtual double consolidate(const std::vector<Neighbour> &ns) const;
    void ready() const;
    void check_query(const DatasetMatrix &x) const;
    double distance(std::span<const double> a, std::size_t i) const;

  public:
    explicit NeighbourModel(std::unique_ptr<IDistance> d);
    void fit(const DatasetMatrix &x, std::span<const double> y) override;
    std::vector<double> predict(const DatasetMatrix &x) const override;
};

// todo: что надо сделать с признаками, чтобы корректно можно было считать расстояние между объектами в KNN и NWReg?

// один код вероятностей для kNN и NW при помощи шаблона Base
template <class Base> class ClassifierModel : public Base, public IClassifier {
    std::vector<double> probabilities(const std::vector<Neighbour> &ns) const;
    double consolidate(const std::vector<Neighbour> &ns) const override;

  public:
    using Base::Base;
    std::span<const double> classes() const override;
    DatasetMatrix predict_proba(const DatasetMatrix &features) const override;
};
