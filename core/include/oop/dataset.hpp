#pragma once
#include <iostream>
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <filesystem>

namespace fs = std::filesystem;

struct NumericColumn {
    std::vector<std::optional<double>> values;
};

struct CategoricalColumn {
    std::vector<std::optional<std::string>> values;
};

using Column = std::variant<NumericColumn, CategoricalColumn>;

class Dataset {
    private:
        int rows = 0;
        //std::string column_types;
        std::vector<std::string> columns_list;
        std::vector<int> missing;
        std::unordered_map<std::string, Column> columns_;

    public:
        Dataset(fs::path path);
        void init_category(std::string str, char category);
        void add_row(std::vector<std::string>& values);
        void add_value(int index, std::string value);
        void show_info();
};





