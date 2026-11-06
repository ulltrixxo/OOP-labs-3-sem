#pragma once
#include "mlcore/matrix.hpp"
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

struct Split;

/** @brief Numeric response or encoded categorical response. */
enum class TargetType {Numeric, Categorical};

/** @brief Responses and, for categorical targets, the label-to-code dictionary. */
struct Target {
    TargetType type;
    std::vector<double> values;
    std::map<std::string, int> categorical_values; 
};
/** @brief Named target columns; iteration order is unspecified. */
using Targets = std::unordered_map<std::string, Target>;

/** @brief Own features and targets with matching row counts. */
class ModelMatrix {
    DatasetMatrix data;
    Targets targets;
    public:
/** @brief Copy features and targets.
 * @throws SizeError On mismatched row counts or categories on a numeric target. */
    ModelMatrix(const DatasetMatrix& data, const Targets& targets);
/** @brief Shuffle row indices with mt19937 and split features and targets together.
 * @param proportion Training fraction in [0,1]; must be finite.
 * @param random_state Seed; reproducible within the same library implementation.
 * @return Training and test copies; training size is round(rows*proportion).
 * @throws BadArgument If proportion is outside [0,1]. */
    Split split(double proportion, unsigned random_state) const;
/** @brief Borrow the feature matrix. */
    const DatasetMatrix& get_data() const;
/** @brief Borrow the target dictionary. */
    const Targets& get_targets() const;
/** @brief Borrow a target column; throws std::out_of_range for an unknown name. */
    std::span<const double> target_values(const std::string& name) const;
/** @brief Load features and targets from a native binary file.
 * @param path Input filename.
 * @throws FileReadError On missing or incomplete input.
 * @note Format depends on ABI and is intended for locally generated files. */
    static ModelMatrix load(
        const std::filesystem::path& path
    );
};

/** @brief Owning training and test subsets. */
struct Split {
    ModelMatrix train;
    ModelMatrix test;
};

/** @brief Save features and all targets in one native binary file.
 * @param data Model data with a feature name for each column.
 * @param path Output filename, overwritten if it exists.
 * @throws std::runtime_error On an output failure. */
void save(const ModelMatrix& data, const std::filesystem::path& path);
