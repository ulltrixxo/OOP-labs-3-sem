#pragma once
#include <iostream>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <filesystem>
#include <span>

namespace fs = std::filesystem;

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

class Dataset {
    private:
        int rows = 0;
        //std::string column_types;
        std::vector<std::string> columns_list;
        std::unordered_map<std::string, int> missing_values;
        std::unordered_map<std::string, Column> columns_;
        DatasetInfo info;
    public:
        Dataset(fs::path path);
        void init_category(std::string category, std::vector<std::string> raw_data);
        void add_row(std::vector<std::string>& values);
        void analyze();
        FeatureInfo calculate(std::string& category);
        void show_info();
        int get_size();
        int get_rows();
        int show_stat(std::size_t n);
};

NumericFeatureInfo calculate_num_arr(std::span<double> array);





