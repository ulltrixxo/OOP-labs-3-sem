#pragma once
#include "mlcore/distance.hpp"
#include "mlcore/matrix.hpp"
#include "mlcore/models/model.hpp"
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

/// Neighbour weighting policy; Distance uses inverse distance.
enum class Weighting { Uniform, Distance };

/// A training row and its distance from the query.
struct Neighbour
{
    double distance; ///< Nonnegative metric result.
    std::size_t index; ///< Original training row index.
};

/** @brief Shared brute-force kNN implementation with O(k) neighbour storage.
 * Owns its metric and copies the training data. Equal distances prefer the
 * smaller training index. A failed validation leaves the previous fit intact.
 */
class kNNBase : public IModel {
    protected:
    std::unique_ptr<IDistance> dist_;
    std::size_t k_;
    Weighting weighting_;
    bool fitted_ = false;
    /// Common relative weights; exact matches exclude all nonzero distances.
    std::vector<double> weights(const std::vector<Neighbour>& neighbours) const;
    DatasetMatrix train_data_; // обучающая выборка
    std::vector<double> target_; // ответы на обучающую выборку
    /** @brief Construct a model with exclusive ownership of its metric.
     * @param k Positive neighbour count; checked against row count in fit.
     * @param distance Non-null owning metric pointer.
     * @param weighting Uniform or inverse-distance weights.
     * @throws BadArgument For zero k, null metric or unknown weighting.
     */
    kNNBase(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting = Weighting::Uniform);
    virtual double consolidate(const std::vector<Neighbour>& neighbours) const = 0;
    public:
    /** @brief Copy a nonempty RowMajor training set and one target per row.
     * @param features Training features; the model keeps a copy.
     * @param target One finite response per training row.
     * @throws BadArgument For empty data, invalid k or nonfinite data/targets.
     * @throws SizeError If target length differs from row count.
     * @throws MatrixTypeError For unsupported layout.
     */
    void fit(
        const DatasetMatrix& features,
        std::span<const double> target
    ) override;
    /** @brief Predict one response per query row; an empty query is allowed.
     * @param features Query rows, not retained after the call.
     * @return One response per query row.
     * @throws MatrixTypeError For non-RowMajor queries.
     * @throws ModelFitError Before successful fit.
     * @throws SizeError For a different feature count.
     * @throws BadArgument For nonfinite features or invalid metric results.
     */
    std::vector<double> predict(
        const DatasetMatrix& features
    ) const override;
};

/// Weighted mean of neighbour targets; exact matches are averaged.
class kNNRegressor : public kNNBase {
    double consolidate(const std::vector<Neighbour>& neighbours) const override;
    public:
    /** @brief Construct a model with exclusive ownership of its metric.
     * @param k Positive neighbour count; checked against row count in fit.
     * @param distance Non-null owning metric pointer.
     * @param weighting Uniform or inverse-distance weights.
     * @throws BadArgument For zero k, null metric or unknown weighting.
     */
    kNNRegressor(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting = Weighting::Uniform);
};

/// Weighted class votes; ties prefer the smaller numeric class label.
class kNNClassifier : public kNNBase {
    double consolidate(const std::vector<Neighbour>& neighbours) const override;
    public:
    /** @brief Construct a model with exclusive ownership of its metric.
     * @param k Positive neighbour count; checked against row count in fit.
     * @param distance Non-null owning metric pointer.
     * @param weighting Uniform or inverse-distance weights.
     * @throws BadArgument For zero k, null metric or unknown weighting.
     */
    kNNClassifier(std::size_t k, std::unique_ptr<IDistance> distance, Weighting weighting = Weighting::Uniform);
};