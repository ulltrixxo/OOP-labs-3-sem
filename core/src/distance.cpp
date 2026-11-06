#include "mlcore/distance.hpp"
#include "mlcore/exceptions.hpp"
#include <cmath>
#include <format>
#include <stdexcept>

double EuclideanDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw SizeError(diagnostics::message(std::format(
        "Cannot compute distance: dimension mismatch (a: {}, b: {})",
        a.size(), b.size())));
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

double ManhattanDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw SizeError(diagnostics::message(std::format(
        "Cannot compute distance: dimension mismatch (a: {}, b: {})",
        a.size(), b.size())));
    double sum = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

double ChebyshevDistance::compute(
        std::span<const double> a,
        std::span<const double> b
) const {
    if (a.size() != b.size()) throw SizeError(diagnostics::message(std::format(
        "Cannot compute distance: dimension mismatch (a: {}, b: {})",
        a.size(), b.size())));
    double sum = 0;
    double res;
    for (std::size_t i = 0; i < a.size(); ++i) {
        res = std::abs(a[i] - b[i]);
        sum = res > sum ? res : sum;
    }
    return sum;
}
