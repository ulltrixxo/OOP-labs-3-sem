#pragma once
#include "mlcore/matrix.hpp"
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <filesystem>
#include <span>

namespace fs = std::filesystem;

struct Config {
    char delimiter;
    std::vector<std::string> missing_values;
    std::vector<std::string> unnecessary_columns;
    std::string impute_missing_value;
    std::vector<std::string> targets;
    bool in_missing_values(const std::string& value) const;
    bool unnecessary_column(const std::string& value) const;
    bool in_targets(const std::string& value) const;
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

    std::map<
        std::string,
        std::size_t
    > frequencies;
};

using FeatureInfo =
    std::variant<
        NumericFeatureInfo,
        CategoricalFeatureInfo
    >;
struct DatasetInfo {
    std::size_t rows;
    std::vector<std::string> columns;
    std::unordered_map<std::string, FeatureInfo> features;
};

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
        Dataset(fs::path path, const Config& config);
        DatasetInfo analyze() const;

        int get_size() const;
        int get_rows() const;
        std::vector<std::string> get_columns_list() const;

        void impute(DatasetInfo& info);
        void scale_numeric_features(DatasetInfo& info);
        void OneHot(DatasetInfo& info);
        ModelMatrix encode(Layout encoding_type = Layout::RowMajor);
};

void save(const DatasetMatrix& data, fs::path path);

// todo: подразбить на файлы, проверить, что Dataset правильно выполняет инкапсуляцию
// убрать волшебные литералы, завести константы для них
