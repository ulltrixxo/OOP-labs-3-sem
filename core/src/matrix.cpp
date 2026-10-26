#include "mlcore/matrix.hpp"
#include "mlcore/dataset.hpp"
#include <fstream>
#include <format>
#include <string>
#include <type_traits>
#include <charconv>
#include <algorithm>
#include <vector>
#include <variant>
#include <cmath>
#include <unordered_set>
#include <stdexcept>
#include <span>
#include <random>
#include <ranges>

DatasetMatrix::DatasetMatrix(std::size_t rows, std::size_t cols, Layout layout,
const std::vector<std::string>& feature_names) : feature_names(feature_names), rows_(rows), cols_(cols), layout_(layout){
    data_.resize(rows*cols);
}
std::size_t DatasetMatrix::get_rows() const {
    return rows_;
}
std::size_t DatasetMatrix::get_cols() const {
    return cols_;
}
Layout DatasetMatrix::get_layout() const {
    return layout_;
}
DatasetMatrix DatasetMatrix::load(
    const std::filesystem::path& path
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening matrix: " + path.string());
    }
    std::size_t rows, cols; Layout layout;
    file.read(reinterpret_cast<char*>(&cols), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&rows), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&layout), sizeof(Layout));
    DatasetMatrix result = DatasetMatrix(rows, cols, layout);
    result.feature_names.reserve(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        std::size_t len;
        file.read(reinterpret_cast<char*>(&len), sizeof(std::size_t));
        std::string str;
        str.resize(len);
        file.read(str.data(), sizeof(char) * len);
        result.feature_names.push_back(str);
    }
    file.read(reinterpret_cast<char*>(result.data_.data()), sizeof(double) * cols * rows);
    return result;
}

std::string_view DatasetMatrix::get_feature_name(std::size_t i) const {
    return feature_names[i];
}

const double* DatasetMatrix::get_data() const {
    return data_.data();
}

const double& DatasetMatrix::operator()(std::size_t row, std::size_t column) const {
    return data_[column + cols_*row];
}

double& DatasetMatrix::operator()(std::size_t row, std::size_t column) {

    return data_[column + cols_*row];
}

std::span<const double> DatasetMatrix::operator[](std::size_t index) const {
    if (layout_ == Layout::ColumnMajor) {throw std::runtime_error("attempt to take element of ColumnMajor matrix");}
    std::span<const double> result(data_.data() + cols_ * index, cols_);
    return result;
}

std::span<double> DatasetMatrix::operator[](std::size_t index) {
    if (layout_ == Layout::ColumnMajor) {throw std::runtime_error("attempt to take element of ColumnMajor matrix");}
    std::span<double> result(data_.data() + cols_ * index, cols_);
    return result;
}

double& DatasetMatrix::at(std::size_t row, std::size_t column) {
    if (row >= rows_ || column >= cols_) throw std::out_of_range("out of range of matrix"); 
    return data_.at(column + cols_ * row);
}

const double& DatasetMatrix::at(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= cols_) throw std::out_of_range("out of range of matrix"); 
    return data_.at(column + cols_ * row);
}

ModelMatrix::ModelMatrix(const DatasetMatrix& data, const Targets& targets) : data(data), targets(targets) {
    for (const auto& [name, target] : targets) {
        if (const auto* numeric = std::get_if<NumericTarget>(&target)) {
            if (numeric->values.size() != data.get_rows()) throw std::runtime_error("ModelMatrix: sizes are not equal");
        } else if (const auto* cat = std::get_if<CategorialTarget>(&target)) {
            if (cat->values.size() != data.get_rows()) throw std::runtime_error("ModelMatrix: sizes are not equal");
        }
    }
}

const std::vector<std::string>& DatasetMatrix::get_feature_names() const {
    return feature_names;
}


/*
    Разделение выборки на:
    - обучающую
    - тестовую
*/
Split ModelMatrix::split(double proportion, unsigned random_state) const {
    if (proportion < 0 || proportion > 1) throw std::runtime_error("Proportion on split must belong [0;1]");

    // делим выборку на:

    std::size_t rows = data.get_rows(), cols = data.get_cols();
    const std::vector<std::string>& feature_names = data.get_feature_names();
    unsigned train_rows = std::round(rows * proportion); // обучающую
    unsigned test_rows = rows - train_rows; // тестовую

    // создаём матрицы
    DatasetMatrix train(train_rows, cols, Layout::RowMajor, feature_names);
    Targets train_targets;
    DatasetMatrix test(test_rows, cols, Layout::RowMajor, feature_names);
    Targets test_targets;

    // std::vector<std::string> targets;
    // for (const std::string& target : targets) {
        
    // }
    std::mt19937 engine(random_state);

    std::vector<unsigned> nums(rows);
    for (std::size_t i = 0; i < nums.size(); ++i) {
        nums[i] = i;
    }

    std::shuffle(nums.begin(), nums.end(), engine);
    

    for (std::size_t i = 0; i < rows; ++i) {
        if (i < train_rows) {
            std::ranges::copy(this->data[nums[i]], train[i].begin());
        }
        else std::ranges::copy(this->data[nums[i]], test[i - train_rows].begin());
    }

    for (const auto& [target, target_data] : targets) {
        if (const auto* numeric = std::get_if<NumericTarget>(&target_data)) {
            std::vector<double>& train_values = train_targets[target].emplace<NumericTarget>().values; 
            train_values.resize(train_rows);
            std::vector<double>& test_values = test_targets[target].emplace<NumericTarget>().values; 
            test_values.resize(test_rows);

            for (std::size_t i = 0; i < rows; ++i) {
                if (i < train_rows) {
                    train_values[i] = numeric->values[nums[i]];
                } else test_values[i - train_rows] = numeric->values[nums[i]];
            }
        } else if (const auto* categorical = std::get_if<CategorialTarget>(&target_data)) {
            CategorialTarget& cat_train_target = train_targets[target].emplace<CategorialTarget>(); 
            CategorialTarget& cat_test_target = test_targets[target].emplace<CategorialTarget>(); 
            cat_train_target.values.resize(train_rows); 
            cat_test_target.values.resize(test_rows);

            for (std::size_t i = 0; i < rows; ++i) {
                if (i < train_rows) {
                    cat_train_target.values[i] = categorical->values[nums[i]];
                } else cat_test_target.values[i - train_rows] = categorical->values[nums[i]];
            } 

            cat_train_target.categorical_values = std::get<CategorialTarget>(target_data).categorical_values;
            cat_test_target.categorical_values = std::get<CategorialTarget>(target_data).categorical_values;
        }
    }
    return {{train, train_targets}, {test, test_targets}};
}

const DatasetMatrix& ModelMatrix::get_data() const {
    return data;
}

const Targets& ModelMatrix::get_targets() const {
    return targets;
}



