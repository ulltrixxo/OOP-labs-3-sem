#pragma once
#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

/** @brief Storage tag; the supported operational path is RowMajor. */
enum class Layout
{
    RowMajor,
    ColumnMajor
};


/** @brief Owning contiguous numeric matrix. Row views borrow its memory. */
class DatasetMatrix
{
private:
    std::vector<std::string> feature_names;
    std::vector<double> data_;

    std::size_t rows_;
    std::size_t cols_;

    Layout layout_;

public:
/** @brief Allocate rows*cols zero values.
 * @param rows Row count.
 * @param cols Feature count.
 * @param layout Use RowMajor for indexing and row views.
 * @param feature_names Optional names; supply one per column for serialization. */
    DatasetMatrix(std::size_t rows = 0, std::size_t cols = 0, Layout layout = Layout::RowMajor,
        const std::vector<std::string>& feature_names = {});
/** @brief Unchecked mutable RowMajor element access. */
    double& operator()(
        std::size_t row,
        std::size_t column
    );

/** @brief Unchecked const RowMajor element access. */
    const double& operator()(
        std::size_t row,
        std::size_t column
    ) const;

/** @brief Number of rows. */
    std::size_t get_rows() const;
/** @brief Number of columns. */
    std::size_t get_cols() const;
/** @brief Stored layout tag. */
    Layout get_layout() const;
/** @brief Borrow a feature name; i must be in the name array. */
    std::string_view get_feature_name(std::size_t i) const;
/** @brief Borrow all feature names. */
    const std::vector<std::string>& get_feature_names() const;
/** @brief Borrow the contiguous numeric storage. */
    const double* get_data() const;
/** @brief Read the matrix prefix from the current stream position.
 * @param file Binary input stream; left immediately before the targets.
 * @throws FileReadError On incomplete input.
 * @throws MatrixTypeError For a layout other than RowMajor. */
    static DatasetMatrix load(
        std::istream& file
    );

/** @brief Borrow a const row; index must be valid.
 * @throws MatrixTypeError For ColumnMajor. */
    std::span<const double> operator[](std::size_t index) const;
/** @brief Borrow a mutable row; index must be valid.
 * @throws MatrixTypeError For ColumnMajor. */
    std::span<double> operator[](std::size_t index);
/** @brief Checked mutable RowMajor access; throws std::out_of_range. */
    double& at(std::size_t row, std::size_t column);
/** @brief Checked const RowMajor access; throws std::out_of_range. */
    const double& at(std::size_t row, std::size_t column) const;


};
