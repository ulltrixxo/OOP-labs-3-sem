#pragma once
#include <vector>
#include <optional>
#include <variant>
#include <unordered_map>
#include <map>
#include <filesystem>
#include <span>



enum class Layout
{
    RowMajor,
    ColumnMajor
};

struct Split;

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
    std::string_view get_feature_name(std::size_t i) const;
    const std::vector<std::string>& get_feature_names() const;
    const double* get_data() const;
    static DatasetMatrix load(
        const std::filesystem::path& path
    );
    
    std::span<const double> operator[](std::size_t index) const;
    std::span<double> operator[](std::size_t index);
    double& at(std::size_t row, std::size_t column);
    const double& at(std::size_t row, std::size_t column) const;


};


struct NumericTarget {
    std::vector<double> values;
    // значения откликов (категориальные кодируются числами)
};

struct CategorialTarget {
    std::vector<double> values;  
    // значения откликов (категориальные кодируются числами)
    std::map<std::string, int> categorical_values; 
    // значения категориальных откликов
    // ОТКЛИК : значение отклика : номер
};

using Target = std::variant<NumericTarget, CategorialTarget>;
using Targets = std::unordered_map<std::string, Target>;

class ModelMatrix {
    DatasetMatrix data;
    Targets targets;
    public:
    ModelMatrix(const DatasetMatrix& data, const Targets& targets);
    Split split(double proportion, unsigned random_state) const;
    const DatasetMatrix& get_data() const;
    const Targets& get_targets() const;
};

struct Split {
    ModelMatrix train;
    ModelMatrix test;
};