#pragma once
#include <span>

/** @brief Distance strategy; models do not depend on concrete metric classes. */
class IDistance {
    public:
/** @brief Enable safe polymorphic destruction. */
    virtual ~IDistance() = default;

/** @brief Compute a distance without retaining input spans.
 * @param a First feature row.
 * @param b Second feature row of the same length.
 * @return Nonnegative distance; zero for two empty rows.
 * @throws SizeError If dimensions differ. */
    virtual double compute (
        std::span<const double> a,
        std::span<const double> b
    ) const = 0;
};

/** @brief Euclidean distance (square root of sum of squared differences). */
class EuclideanDistance : public IDistance {
    public:
    /** @brief Compute a distance without retaining input spans.
 * @param a First feature row.
 * @param b Second feature row of the same length.
 * @return Nonnegative distance; zero for two empty rows.
 * @throws SizeError If dimensions differ. */
    virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};

/** @brief Sum of absolute coordinate differences. */
class ManhattanDistance : public IDistance {
    public:
    /** @brief Compute a distance without retaining input spans.
 * @param a First feature row.
 * @param b Second feature row of the same length.
 * @return Nonnegative distance; zero for two empty rows.
 * @throws SizeError If dimensions differ. */
    virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};

/** @brief Maximum absolute coordinate difference. */
class ChebyshevDistance : public IDistance {
    public:
    /** @brief Compute a distance without retaining input spans.
 * @param a First feature row.
 * @param b Second feature row of the same length.
 * @return Nonnegative distance; zero for two empty rows.
 * @throws SizeError If dimensions differ. */
    virtual double compute (
        std::span<const double> a,
        std::span<const double> b
        ) const;
};
