#pragma once
#include <vector>
#include <string>

/** @brief CSV parsing and preprocessing options. */
struct Config {
    char delimiter;
    std::vector<std::string> missing_values;
    std::vector<std::string> unnecessary_columns;
    std::string impute_missing_value;
    std::vector<std::string> targets;
/** @brief Test exact membership in the missing-token list. */
    bool in_missing_values(const std::string& value) const;
/** @brief Test whether a column should be dropped. */
    bool unnecessary_column(const std::string& value) const;
/** @brief Test whether a column is a response. */
    bool in_targets(const std::string& value) const;
};