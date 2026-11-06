#include "mlcore/model_matrix.hpp"
#include "mlcore/exceptions.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
#include <format>

ModelMatrix::ModelMatrix(const DatasetMatrix& data, const Targets& targets) : data(data), targets(targets) {
    for (const auto& [name, target] : targets) {
        if (target.values.size() != data.get_rows()) {
            throw SizeError(diagnostics::message(std::format(
                "Target '{}': the value count ({}) does not match the matrix row count ({})",
                name, target.values.size(), data.get_rows())));
        }
        if (target.type == TargetType::Numeric && !target.categorical_values.empty()) {
            throw SizeError(diagnostics::message(std::format(
                "Numeric target '{}' contains a category table ({} entries)",
                name, target.categorical_values.size())));
        }
    }
}

/*
    Разделение выборки на:
    - обучающую
    - тестовую
*/
Split ModelMatrix::split(double proportion, unsigned random_state) const {
    if (proportion < 0 || proportion > 1) throw BadArgument(diagnostics::message(std::format(
        "The training proportion must be in [0, 1]; got {}", proportion)));

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

    std::mt19937 engine(random_state); // генератор рандомных чисел

    std::vector<unsigned> nums(rows);
    for (std::size_t i = 0; i < nums.size(); ++i) {
        nums[i] = i;
    }

    std::shuffle(nums.begin(), nums.end(), engine); // перемешать массив
    

    for (std::size_t i = 0; i < rows; ++i) {
        if (i < train_rows) {
            std::ranges::copy(this->data[nums[i]], train[i].begin());
        }
        else std::ranges::copy(this->data[nums[i]], test[i - train_rows].begin());
    }

    for (const auto& [target, target_data] : targets) {
        // разделение по размерам по пропорции
        train_targets[target].values.resize(train_rows);
        test_targets[target].values.resize(test_rows);

        train_targets[target].type = target_data.type;
        test_targets[target].type = target_data.type;

        // копирование значений
        for (std::size_t i = 0; i < rows; ++i) {
            if (i < train_rows) {
                train_targets[target].values[i] = target_data.values[nums[i]];
            } else test_targets[target].values[i - train_rows] = target_data.values[nums[i]];
        }

        // перенос категориальных значений
        // если признак числовой, то просто скопируется пустой словарь
        train_targets[target].categorical_values = target_data.categorical_values;
        test_targets[target].categorical_values = target_data.categorical_values;
    }
    return {{train, train_targets}, {test, test_targets}};
}

const DatasetMatrix& ModelMatrix::get_data() const { return data; }
const Targets& ModelMatrix::get_targets() const { return targets; }
std::span<const double> ModelMatrix::target_values(const std::string& name) const { return targets.at(name).values; }

void save(const ModelMatrix& data, const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw FileSaveError(diagnostics::message(
            std::format("Cannot open file '{}' for writing", path.string())));
    }
    DatasetMatrix datasetmatrix_data = data.get_data();
    Targets datasetmatrix_targets = data.get_targets();

    // запись чисел
    const auto write_number = [&file, &path](const auto& value) {
        file.write(reinterpret_cast<const char*>(&value),
                   sizeof(value));

        if (!file) {
            throw FileSaveError(diagnostics::message(std::format(
                "File '{}': cannot write a number ({} bytes)", path.string(), sizeof(value))));
        }
    };

    // Строка: количество байт, затем сами байты.
    const auto write_string = [&file, &path, &write_number](const std::string& value) {
        const std::size_t length = value.size();
        write_number(length);

        file.write(value.data(), length);

        if (!file) { throw std::runtime_error(diagnostics::message(std::format(
            "File '{}': cannot write a string ({} bytes)", path.string(), length))); }

    };

    std::size_t cols = datasetmatrix_data.get_cols(), rows = datasetmatrix_data.get_rows();
    Layout layout = datasetmatrix_data.get_layout();
    const double* datasetmatrix_ptr = datasetmatrix_data.get_data();
    file.write(reinterpret_cast<const char*>(&cols), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&layout), sizeof(Layout));
    for (std::size_t i = 0; i < datasetmatrix_data.get_cols(); ++i) {
        std::size_t len = datasetmatrix_data.get_feature_name(i).size();
        file.write(reinterpret_cast<const char*>(&len), sizeof(std::size_t));
        for (char c : datasetmatrix_data.get_feature_name(i)) {
            file.write(&c, sizeof(char));
        }
    }
    file.write(reinterpret_cast<const char*>(datasetmatrix_ptr), rows * cols * sizeof(double));

    write_number(datasetmatrix_targets.size());

    for (const auto& [name, target] : datasetmatrix_targets) {
        write_string(name);

        const std::size_t type = static_cast<std::size_t>(target.type);
        write_number(type);

        write_number(target.values.size());
        for (const double value : target.values) {
            write_number(value);
        }

        write_number(target.categorical_values.size());
        for (const auto& [label, code] : target.categorical_values) {
            write_string(label);
            write_number(code);
        }
    }

    file.close();
    if (!file) {
        throw FileSaveError(diagnostics::message(
            std::format("Cannot finish writing and close file '{}'", path.string())));
    }
}

ModelMatrix ModelMatrix::load(
    const std::filesystem::path& path
) {
    std::ifstream file(path, std::ios::binary);

    // сначала идёт считывание матрицы
    DatasetMatrix matrix = DatasetMatrix::load(file);


    // объявление лямбда-функций 
    // прочитать число
    const auto read_number = [&file, &path](auto& value) {
        file.read(reinterpret_cast<char*>(&value), sizeof(value));

        if (!file) {
            throw FileReadError(diagnostics::message(std::format(
                "File '{}': cannot read a number ({} bytes required); the file is truncated or a read error occurred",
                path.string(), sizeof(value))));
        }
    };

    // прочитать строку
    const auto read_string = [&file, &path, &read_number]() {
        std::size_t length = 0;
        read_number(length);

        std::string value(length, '\0');
        file.read(value.data(), length);

        if (!file) {
            throw FileReadError(diagnostics::message(std::format(
                "File '{}': cannot read a string of {} bytes; the file is truncated or a read error occurred",
                path.string(), length)));
        }

        return value;
    };


    // считывание откликов
    Targets targets;

    std::size_t target_count = 0;
    read_number(target_count);

    for (std::size_t i = 0; i < target_count; ++i) {
        const std::string name = read_string();
        Target target{};

        std::size_t type = 0;
        read_number(type);

        if (type != static_cast<std::size_t>(TargetType::Numeric) &&
            type != static_cast<std::size_t>(TargetType::Categorical)) {
            throw FileReadError(diagnostics::message(std::format(
                "File '{}', target '{}': unknown type code {}", path.string(), name, type)));
        }
        target.type = static_cast<TargetType>(type);

        std::size_t value_count = 0;
        read_number(value_count);

        target.values.resize(value_count);
        for (double& value : target.values) {
            read_number(value);
        }

        std::size_t category_count = 0;
        read_number(category_count);

        for (std::size_t j = 0; j < category_count; ++j) {
            const std::string label = read_string();

            int code = 0;
            read_number(code);

            if (!target.categorical_values.emplace(label, code).second) {
                throw FileReadError(diagnostics::message(std::format(
                    "File '{}', target '{}': duplicate category '{}'", path.string(), name, label)));
            }
        }

        if (target.type == TargetType::Numeric && !target.categorical_values.empty()) {
            throw FileReadError(diagnostics::message(std::format(
                "File '{}': numeric target '{}' contains a category table", path.string(), name)));
        }

        if (!targets.emplace(name, std::move(target)).second) {
            throw FileReadError(diagnostics::message(std::format(
                "File '{}': duplicate target '{}'", path.string(), name)));
        }
    }

    return ModelMatrix(matrix, targets);
}