#include "mlcore/matrix.hpp"
#include "mlcore/exceptions.hpp"
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
std::size_t DatasetMatrix::get_rows() const { return rows_; }
std::size_t DatasetMatrix::get_cols() const { return cols_; }
Layout DatasetMatrix::get_layout() const { return layout_; }

DatasetMatrix DatasetMatrix::load(
    std::istream& file
) {
    if (!file) {
        throw FileReadError(diagnostics::message(
            "Cannot read the matrix: the input stream is in an error state"));
    }
    std::size_t rows = 0, cols = 0; Layout layout{};
    file.read(reinterpret_cast<char*>(&cols), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&rows), sizeof(std::size_t));
    file.read(reinterpret_cast<char*>(&layout), sizeof(Layout));
    if (!file) throw FileReadError(diagnostics::message("Truncated matrix header"));
    if (layout != Layout::RowMajor) throw MatrixTypeError(diagnostics::message("Binary loading requires RowMajor storage"));
    if (cols != 0 && rows > std::vector<double>().max_size() / cols) throw FileReadError(diagnostics::message("Matrix dimensions are too large"));

    DatasetMatrix result = DatasetMatrix(rows, cols, layout);
    result.feature_names.reserve(cols);
    for (std::size_t i = 0; i < cols; ++i) {
        std::size_t len = 0;
        file.read(reinterpret_cast<char*>(&len), sizeof(std::size_t));
        if (!file) throw FileReadError(diagnostics::message("Truncated feature-name length"));
        std::string str;
        str.resize(len);
        file.read(str.data(), sizeof(char) * len);
        if (!file) throw FileReadError(diagnostics::message("Truncated feature name"));
        result.feature_names.push_back(str);
    }
    file.read(reinterpret_cast<char*>(result.data_.data()), sizeof(double) * cols * rows);
    if (!file) throw FileReadError(diagnostics::message("Truncated matrix values"));
    return result;
}

std::string_view DatasetMatrix::get_feature_name(std::size_t i) const { return feature_names[i]; }
const double* DatasetMatrix::get_data() const { return data_.data(); }
const double& DatasetMatrix::operator()(std::size_t row, std::size_t column) const { return data_[column + cols_*row]; }
double& DatasetMatrix::operator()(std::size_t row, std::size_t column) { return data_[column + cols_*row]; }
const std::vector<std::string>& DatasetMatrix::get_feature_names() const { return feature_names; }

std::span<const double> DatasetMatrix::operator[](std::size_t index) const {
    if (layout_ == Layout::ColumnMajor) {throw MatrixTypeError(diagnostics::message("Cannot obtain a row as a span: the matrix uses ColumnMajor storage"));}
    std::span<const double> result(data_.data() + cols_ * index, cols_);
    return result;
}

std::span<double> DatasetMatrix::operator[](std::size_t index) {
    if (layout_ == Layout::ColumnMajor) {throw MatrixTypeError(diagnostics::message("Cannot obtain a row as a span: the matrix uses ColumnMajor storage"));}
    std::span<double> result(data_.data() + cols_ * index, cols_);
    return result;
}

double& DatasetMatrix::at(std::size_t row, std::size_t column) {
    if (row >= rows_ || column >= cols_) throw std::out_of_range(diagnostics::message(std::format(
        "Index ({}, {}) is outside the matrix ({} rows x {} columns)",
        row, column, rows_, cols_
    )));
    return data_.at(column + cols_ * row);
}

const double& DatasetMatrix::at(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= cols_) throw std::out_of_range(diagnostics::message(std::format(
        "Index ({}, {}) is outside the matrix ({} rows x {} columns)",
        row, column, rows_, cols_
    )));
    return data_.at(column + cols_ * row);
}


// todo: рассказать, в чем заключается полиморфизм, обеспечиваемый шаблонами. Сравнить виртуальные функции и шаблоны с точки зрения обеспечения полиморфного поведения
// раскрутка стека
