#pragma once
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#define HISTOGRAM_COUNT 10
/** @brief Numeric summary: missing count, moments, quantiles and ten histogram bins. */
struct NumericFeatureInfo {
    int missing_count;

    double min;
    double max;

    double mean;
    double variance;

    double median;

    double q05;
    double q25;
    double q75;
    double q95;

    std::pair<double, double> ranges[HISTOGRAM_COUNT];
    int values[HISTOGRAM_COUNT];
};

/** @brief Missing count and frequencies of present categories. */
struct CategoricalFeatureInfo
{
    int missing_count;

    std::map<
        std::string,
        std::size_t
    > frequencies;
};

/** @brief Numeric or categorical statistics. */
using FeatureInfo =
    std::variant<
        NumericFeatureInfo,
        CategoricalFeatureInfo
    >;

/** @brief Dataset row count, ordered names and per-column statistics. */
struct DatasetInfo {
    std::size_t rows;
    std::vector<std::string> columns;
    std::unordered_map<std::string, FeatureInfo> features;
};
