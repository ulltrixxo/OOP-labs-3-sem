#pragma once
#include "mlcore/config.hpp"
#include "mlcore/feature_info.hpp"
#include "mlcore/matrix.hpp"
#include "mlcore/model_matrix.hpp"
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
constexpr int NUMERIC_UNIQUE_VALUES_LIMIT = 10;
#define OBJECTS_PER_SYMBOL 10   
/** @brief Numeric column; nullopt represents a missing value. */
struct NumericColumn {
    std::vector<std::optional<double>> values;
};

/** @brief String column; nullopt represents a missing value. */
struct CategoricalColumn {
    std::vector<std::optional<std::string>> values;
};

/** @brief Type-safe numeric or categorical column storage. */
using Column = std::variant<NumericColumn, CategoricalColumn>;

// основной класс датасет
/** @brief CSV data and in-place preprocessing; configured targets are not transformed. */
class Dataset {
    private:
        std::size_t rows = 0, cols = 0;
        //std::string column_types;
        std::vector<std::string> columns_list;
        std::unordered_map<std::string, int> missing_values;
        std::unordered_map<std::string, Column> columns_;
        std::vector<int> skipped_columns; 
        Config config_;

        // приватные методы
        void init_category(const std::string& category, const std::vector<std::string>& raw_data);
        void add_row(std::vector<std::string>& values);
        bool skipped_column(int index) const; 
        FeatureInfo calculate(const std::string& category) const;

        NumericFeatureInfo calculate_num_arr(std::span<double> array) const;
        CategoricalFeatureInfo calculate_categorical_arr(const std::string& category) const;
        bool has_all_targets(std::size_t row) const;
    public:
/** @brief Load CSV using the supplied configuration.
 * @param path CSV file.
 * @param config Delimiter, missing tokens, ignored columns and targets.
 * @throws FileOpenError If opening fails.
 * @throws FileFormatError On invalid header or row shape. */
        Dataset(std::filesystem::path path, const Config& config);
/** @brief Compute statistics for current columns without changing the data. */
        DatasetInfo analyze() const;

/** @brief Return the stored column count. */
        int get_cols() const;
/** @brief Return the stored row count. */
        int get_rows() const;
/** @brief Return a copy of the current ordered column names. */
        std::vector<std::string> get_columns_list() const;

/** @brief Fill feature gaps with numeric medians or the configured category.
 * @param info Statistics from analyze; targets are skipped. */
        void impute(DatasetInfo& info);
/** @brief Standardize numeric features using supplied mean and variance.
 * @param info Statistics for the feature columns.
 * @pre Gaps are filled and numeric variances are positive. */
        void scale_numeric_features(DatasetInfo& info);
/** @brief Replace categorical features with indicator columns, leaving targets intact.
 * @param info Category frequencies and missing counts from analyze. */
        void OneHot(DatasetInfo& info);
/** @brief Copy numeric features and extract targets, dropping rows with missing targets.
 * @param encoding_type Use RowMajor; ColumnMajor is not implemented.
 * @pre Feature gaps are filled and categories have been one-hot encoded. */
        ModelMatrix encode(Layout encoding_type = Layout::RowMajor);
};

// todo: подразбить на файлы, проверить, что Dataset правильно выполняет инкапсуляцию
// убрать волшебные литералы, завести константы для них
