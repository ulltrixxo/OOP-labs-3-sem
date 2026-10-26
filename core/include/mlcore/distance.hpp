#pragma once
#include <span>
#include <memory>
#include "dataset.hpp"

class IDistance {
    public:
    virtual ~IDistance() = default;

    virtual double compute (
        std::span<const double> a,
        std::span<const double> b
    ) const = 0;
};

class EuclideanDistance : public IDistance {
    public:
        virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};

class ManhattanDistance : public IDistance {
    public:
        virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};

class ChebyshevDistance : public IDistance {
    public:
        virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};

struct Neighbour
{
    double distance;
    std::size_t index;
};

class IModel {
    public:
    virtual ~IModel() = default;

    virtual void fit(
        const DatasetMatrix& features,
        std::span<const double> target
    ) = 0; // признаки + ответы; модель учится на этих данных

    virtual std::vector<double> predict(
        const DatasetMatrix& features
    ) const = 0; // выдаёт ответы
};

class kNNBase : public IModel {
    protected:
    std::unique_ptr<IDistance> dist_;
    std::size_t k_;
    DatasetMatrix train_data_;
    std::vector<double> target_;
    kNNBase(std::size_t k, std::unique_ptr<IDistance> distance);
};

class kNNRegressor : public kNNBase {
    public:
    kNNRegressor();
    void fit(        
        const DatasetMatrix& features,
        std::span<const double> target
    );
    std::vector<double> predict(
        const DatasetMatrix& features
    ) const = 0;
};

class kNNClassifier : public kNNBase {

};