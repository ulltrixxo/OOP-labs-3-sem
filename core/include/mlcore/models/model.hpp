#pragma once
#include "mlcore/matrix.hpp"
#include <span>
#include <vector>

/** @brief Common polymorphic interface, independent of the learning algorithm. */
class IModel {
    public:
/** @brief Allow destruction through a base-class pointer. */
    virtual ~IModel() = default;

/** @brief Learn from features and their targets.
 * @param features Training rows.
 * @param target One response per row. Implementations define validation rules. */
    virtual void fit(
        const DatasetMatrix& features,
        std::span<const double> target
    ) = 0; // признаки + ответы; модель учится на этих данных

/** @brief Return one prediction per row.
 * @param features Query rows with the training feature count.
 * @return Predictions in query order. */
    virtual std::vector<double> predict(
        const DatasetMatrix& features
    ) const = 0; // выдаёт ответы
};
