#pragma once
#include <iostream>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <filesystem>
#include <span>

namespace fs = std::filesystem;

struct Config {
    char delimiter;
    std::vector<std::string> missing_value;
    std::vector<std::string> unnecessary_columns;
    std::string impute_missing_value;
};
struct NumericColumn {
    std::vector<std::optional<double>> values;
};

struct CategoricalColumn {
    std::vector<std::optional<std::string>> values;
};

using Column = std::variant<NumericColumn, CategoricalColumn>;

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

    std::pair<double, double> ranges[10];
    int values[10];
};

struct CategoricalFeatureInfo
{
    int missing_count;

    std::unordered_map<
        std::string,
        std::size_t
    > frequencies;
};

using FeatureInfo =
    std::variant<
        NumericFeatureInfo,
        CategoricalFeatureInfo
    >;
using DatasetInfo =
    std::unordered_map<std::string, FeatureInfo>;

enum class Layout
{
    RowMajor,
    ColumnMajor
};

class DatasetMatrix
{
private:
    std::vector<std::string> feature_names;
    std::vector<double> data_;

    std::size_t rows_;
    std::size_t cols_;

    Layout layout_;

public:
    DatasetMatrix(std::size_t rows = 0, std::size_t cols = 0, Layout layout = Layout::RowMajor, 
        const std::vector<std::string>& feature_names = {});
    double& operator()(
        std::size_t row,
        std::size_t column
    );

    const double& operator()(
        std::size_t row,
        std::size_t column
    ) const;

    std::size_t get_rows() const;
    std::size_t get_cols() const;
    Layout get_layout() const;
    std::string_view get_feature_name(std::size_t i);
    const double* get_data() const;
    static DatasetMatrix load(
        const std::filesystem::path& path
    );
};
class Dataset {
    private:
        int rows = 0;
        //std::string column_types;
        std::vector<std::string> columns_list;
        std::unordered_map<std::string, int> missing_values;
        std::unordered_map<std::string, Column> columns_;
        std::vector<int> skipped_columns; 
        Config config_;
    public:
        Dataset(fs::path path, const Config& config);
        void init_category(const std::string& category, const std::vector<std::string>& raw_data);
        void add_row(std::vector<std::string>& values);
        DatasetInfo analyze() const;
        FeatureInfo calculate(const std::string& category) const;
        void show_info() const;
        int get_size() const;
        int get_rows() const;
        int show_stat(const DatasetInfo& info, std::size_t n) const;
        bool in_missing_values(const std::string& value) const;
        bool unnecessary_column(const std::string& value) const;
        bool skipped_column(int index) const; 
        void impute(DatasetInfo& info);
        void scale_numeric_features(DatasetInfo& info);
        void OneHot(DatasetInfo& info);
        DatasetMatrix encode(Layout encoding_type = Layout::RowMajor);
};

NumericFeatureInfo calculate_num_arr(std::span<double> array);
void save(DatasetMatrix& data);

// todo: подразбить на файлы, проверить, что Dataset правильно выполняет инкапсуляцию
// убрать волшебные литералы, завести константы для них
