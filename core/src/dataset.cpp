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
#include <ranges>
#include <stdexcept>
#include <random>
namespace{
std::optional<double> parse_double(std::string_view s) {
    double value = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size()) return std::nullopt;
    return value;
}
}


Dataset::Dataset(fs::path path, const Config& config) : config_(config) {
    std::vector<std::vector<std::string>> raw_data;
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("Error opening file: " + path.string());
    }

    std::string data, header, value; 
    std::vector<std::string> values;

    // считывание первой строки датасета --- названия признаков.
    int column_index = 0;
    if(!std::getline(file, data)) {
        throw std::runtime_error("file is empty or this is directory");
    }
    std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
    while (std::getline(stream_data, value, config_.delimiter)) {    
        if (config_.unnecessary_column(value)) {
            skipped_columns.push_back(column_index);
            ++column_index;
            continue;
        };// Пропуск ID
        columns_list.push_back(value);
        missing_values[value] = 0;
        ++column_index;
    }
    if (columns_list.empty()) {
        throw std::runtime_error("headers are empty");
    }
    data.clear();
    raw_data.resize(columns_list.size());

    // заполнение сырыми данными
    // В каждой строке полей столько же, скольк5о в заголовке, включая пропускаемые.
    const std::size_t expected_fields = columns_list.size() + skipped_columns.size();
    std::size_t line_number = 1; // строка 1 — заголовок, нумерация как в редакторе

    while (std::getline(file, data)) {
        ++line_number;

        // Полей на одно больше, чем разделителей. Проверка до разбора строки:
        // ловит и недостающие, и лишние поля.
        const auto fields = static_cast<std::size_t>(std::ranges::count(data, config_.delimiter)) + 1;
        if (fields != expected_fields) {
            throw std::runtime_error(std::format(
                "{}: line {}: expected {} values, got {}",
                path.string(), line_number, expected_fields, fields));
        }

        std::size_t file_column = 0;
        std::size_t dataset_column = 0;

        for (const auto field : data | std::views::split(config_.delimiter)) {
            if (!skipped_column(file_column)) {
                raw_data[dataset_column].emplace_back(field.begin(), field.end());
                ++dataset_column;
            }
            ++file_column;
        }
    }
    rows = raw_data[0].size();
    cols = columns_list.size();
    if (rows == 0) {
        throw std::runtime_error("the headers exist, but there is no data.");
    }

    // инициализация категорий датасета на основе сырых данных
    for (std::size_t i = 0; i < columns_list.size(); ++i) {
        init_category(columns_list[i], raw_data[i]);
    }

    // преобразование сырых данных в готовых датасет
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns_list.size(); ++column) {
            if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[column]])) {
                auto& cat_arr = std::get<CategoricalColumn>(columns_[columns_list[column]]);
                if (raw_data[column][row] == "" || config_.in_missing_values(raw_data[column][row])) {
                    cat_arr.values.push_back(std::nullopt);
                    ++missing_values[columns_list[column]];
                    continue;
                }
                cat_arr.values.push_back(raw_data[column][row]);
            } else {
                auto& num_arr = std::get<NumericColumn>(columns_[columns_list[column]]);
                if (config_.in_missing_values(raw_data[column][row])) {
                    num_arr.values.push_back(std::nullopt);
                    ++missing_values[columns_list[column]];
                    continue;
                }
                std::string_view value_ptr = raw_data[column][row];
                num_arr.values.push_back(parse_double(value_ptr));
            }
        }
    }
}

bool Config::in_missing_values(const std::string& value) const {
    for (const std::string& missing_value : missing_values) {
        if (value == missing_value) return true;
    }
    return false;
}

bool Config::unnecessary_column(const std::string& value) const {
    for (const std::string& missing_value : unnecessary_columns) {
        if (value == missing_value) return true;
    }
    return false;
}

bool Config::in_targets(const std::string& value) const {
    for (const std::string& target : targets) {
        if (value == target) return true;
    }
    return false;
}
bool Dataset::skipped_column(int index) const {
    for (int i : skipped_columns) {
        if (index == i) return true;
    }
    return false;
}
// количество столбцов
int Dataset::get_size() const {
    return columns_.size();
}

// количество строк
int Dataset::get_rows() const {
    return rows;
}

void Dataset::init_category(const std::string& category, const std::vector<std::string>& raw_data) {
// инициализация категории на основе сырых данных
// если не получается перевести в число --- категориальный признак
// если получается перевести в число и уникальных значений > 10 --- числовой признак
// в противном случае --- категориальный признак
    double num = 0;
    std::unordered_set<std::string> unique_values;
    for (std::size_t i = 0; i < raw_data.size(); ++i) {
        if (!config_.in_missing_values(raw_data[i])) {
            auto result = std::from_chars(raw_data[i].data(), raw_data[i].data() + raw_data[i].size(), num);
            if (result.ec != std::errc{}) {
                columns_[category].emplace<CategoricalColumn>();
                return;
            }
            unique_values.insert(std::to_string(num));
            if (unique_values.size() > 10) {
                columns_[category].emplace<NumericColumn>();
            return;
            }
        }
    }
    columns_[category].emplace<CategoricalColumn>();
    return;
}

// добавить строку в датасет
void Dataset::add_row(std::vector<std::string>& values) {
    double num = 0.0;
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        if (std::holds_alternative<NumericColumn>(columns_[columns_list[i]])){
            auto& numeric_list = std::get<NumericColumn>(columns_[columns_list[i]]);
            if (config_.in_missing_values(values[i])) {
                numeric_list.values.push_back(std::nullopt);
                ++missing_values[columns_list[i]];
                continue;
            }
            std::string_view values_ptr(values[i].data(), values[i].size());
            std::from_chars(values_ptr.data(), values_ptr.data() + values_ptr.size(), num);
            numeric_list.values.push_back(num);
        } else if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[i]])) {
            auto& categorial_list = std::get<CategoricalColumn>(columns_[columns_list[i]]);
            if (values[i] == "" || config_.in_missing_values(values[i])) {
                ++missing_values[columns_list[i]];
                categorial_list.values.push_back(std::nullopt);
                continue;
            }
            categorial_list.values.push_back(values[i]);
        }
    }
    ++rows;
}

// анализ каждого столбца датасета
DatasetInfo Dataset::analyze() const {
    DatasetInfo info;
    info.columns = columns_list;
    info.rows = rows;
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        info.features[columns_list[i]] = calculate(columns_list[i]);
    }
    return info;
}

NumericFeatureInfo Dataset::calculate_num_arr(std::span<double> array) const {
    std::sort(array.begin(), array.end());
    NumericFeatureInfo result{};
    double sum = 0;
    double squared_sum = 0;
    double min_st, max_st;
    std::size_t n = 0;

    for (std::size_t i = 0; i < array.size(); ++i) {
        sum += array[i];
    }

    result.min = array[0];
    result.max = array[array.size() - 1];
    result.q05 = array[std::trunc(array.size() * 0.05 - 1)];
    result.q25 = array[std::trunc(array.size() * 0.25 - 1) ];
    result.median = array[std::trunc(array.size() * 0.5 - 1)];
    result.q75 = array[std::trunc(array.size() * 0.75 - 1)];
    result.q95 = array[std::trunc(array.size() * 0.95 - 1)];
    result.mean = sum / array.size();
    for (std::size_t i = 0; i < array.size(); ++i) {
        squared_sum += (array[i] - result.mean) * (array[i] - result.mean);
    }
    result.variance = squared_sum / (array.size() - 1);
    double step = (result.max - result.min) / 10.;
    min_st = result.min;
    for (int i = 0; i < 10; ++i) {
        max_st = result.min + step * (i + 1);
        result.ranges[i].first = min_st;
        result.ranges[i].second = max_st;

        while (n < array.size() && min_st <= array[n] && (i != 9 ? array[n] < max_st : array[n] <= max_st)) {
            ++result.values[i];
            ++n;
        }
        min_st = max_st;
    }
    return result;
}

CategoricalFeatureInfo Dataset::calculate_categorical_arr(const std::string& category) const {
    CategoricalFeatureInfo result{};
    auto& categorical_arr = std::get<CategoricalColumn>(columns_.at(category)).values;
    for (std::size_t i = 0; i < categorical_arr.size(); ++i) {
        if (!categorical_arr[i].has_value()) {continue;}
        ++result.frequencies[categorical_arr[i].value()];
    }
    result.missing_count = missing_values.at(category);

    return result;
}
// высчитывание данных о каждом столбце датасета
FeatureInfo Dataset::calculate(const std::string& category) const {
    FeatureInfo statistic;
    if (std::holds_alternative<NumericColumn>(columns_.at(category))) {
        auto& numeric_arr = std::get<NumericColumn>(columns_.at(category)).values;
        auto& numeric_stat = statistic.emplace<NumericFeatureInfo>();
        std::vector<double>sorted_array;

        sorted_array.reserve(numeric_arr.size());
        for(auto i = 0UL; i < numeric_arr.size(); ++i) {
            if (!numeric_arr[i].has_value()) {
                continue;
            }
            sorted_array.push_back(numeric_arr[i].value()); 
        }

        numeric_stat = calculate_num_arr(sorted_array);
        numeric_stat.missing_count = missing_values.at(category);

    } else if (std::holds_alternative<CategoricalColumn>(columns_.at(category))) {
        statistic = calculate_categorical_arr(category);
    }

    return statistic;
}

// показать статистику столбца
void Dataset::impute(DatasetInfo& info) {
        for (const std::string& column : columns_list) {
            // Если признак является откликом, с ним никаких
            // действий не осуществляется
            if (config_.in_targets(column)) continue;
            if (std::holds_alternative<NumericColumn>(columns_[column])) {
                auto& num_arr = std::get<NumericColumn>(columns_[column]).values;
                auto& num_stat = std::get<NumericFeatureInfo>(info.features.at(column));
                for (std::optional<double>& x : num_arr) {
                    if (!x.has_value()) {
                        x = num_stat.median;
                    }
                }
            } else if (std::holds_alternative<CategoricalColumn>(columns_[column])) {
                auto& cat_arr = std::get<CategoricalColumn>(columns_[column]).values;
                for (std::optional<std::string>& x : cat_arr) {
                    if (!x.has_value()) {
                        x = config_.impute_missing_value;
                    }
                }
            }
        }
}

void Dataset::scale_numeric_features(DatasetInfo& info) {
    for (const std::string& column : columns_list) {
        // Если признак является откликом, с ним никаких
        // действий не осуществляетсяы
        if (config_.in_targets(column)) continue;
        if (std::holds_alternative<NumericColumn>(columns_[column])) {
            auto& num_arr = std::get<NumericColumn>(columns_[column]).values;
            auto& num_stat = std::get<NumericFeatureInfo>(info.features[column]);
            for (std::optional<double>& x : num_arr) {
                x = (x.value() - num_stat.mean) / sqrt(num_stat.variance);
            }
        } 
    }
}

void Dataset::OneHot(DatasetInfo& info) {
    std::vector<std::string> new_column_list; // массив с новыми столбцами
    std::string onehot_header;
    for (const std::string& column : columns_list) {
        // Если признак является откликом, с ним никаких
        // действий не осуществляетсяы
        if (config_.in_targets(column)) {
            new_column_list.push_back(column);
            continue; 
        }
        if (std::holds_alternative<CategoricalColumn>(columns_[column])) {
            std::vector<std::string> onehot_headers;
            std::vector<std::string> onehot_values;
            auto& cat_arr = std::get<CategoricalColumn>(columns_[column]).values;
            auto& cat_stat = std::get<CategoricalFeatureInfo>(info.features[column]);
            for (const auto& unique_value : cat_stat.frequencies) {
                onehot_header = column + "=" + unique_value.first;
                columns_[onehot_header].emplace<NumericColumn>().values.resize(rows);
                onehot_headers.push_back(onehot_header);
                onehot_values.push_back(unique_value.first);
            }
            std::sort(onehot_headers.begin(), onehot_headers.end());
            std::sort(onehot_values.begin(), onehot_values.end());
            if (cat_stat.missing_count > 0){
                onehot_headers.push_back(column + "=" + config_.impute_missing_value);
                onehot_values.push_back(config_.impute_missing_value);
                columns_[column + "=" + config_.impute_missing_value].emplace<NumericColumn>().values.resize(rows);
            } 
            for (std::size_t row = 0; row < rows; ++row) {
                for (size_t i = 0; i < onehot_headers.size(); ++i) {
                    std::get<NumericColumn>(columns_[onehot_headers[i]]).values[row] = (cat_arr[row] == onehot_values[i]) ? 1 : 0;
                }
            }
            columns_.erase(column);
            for (std::size_t i = 0; i < onehot_headers.size(); ++i) {
                new_column_list.push_back(onehot_headers[i]);
            }

        } else if (std::holds_alternative<NumericColumn>(columns_[column])) {
            new_column_list.push_back(column);
        }
    }
    columns_list = new_column_list;
}
// todo: рассказать, в чем заключается полиморфизм, обеспечиваемый шаблонами. Сравнить виртуальные функции и шаблоны с точки зрения обеспечения полиморфного поведения
// раскрутка стека
void save(const DatasetMatrix& data, fs::path path) {
    std::ofstream file(path, std::ios::binary);
    std::size_t cols = data.get_cols(), rows = data.get_rows();
    Layout layout = data.get_layout();
    const double* DatasetMatrix_data = data.get_data();
    file.write(reinterpret_cast<const char*>(&cols), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(std::size_t));
    file.write(reinterpret_cast<const char*>(&layout), sizeof(Layout));
    for (std::size_t i = 0; i < data.get_cols(); ++i) {
        std::size_t len = data.get_feature_name(i).size();
        file.write(reinterpret_cast<const char*>(&len), sizeof(std::size_t));
        for (char c : data.get_feature_name(i)) {
            file.write(&c, sizeof(char));
        }
    }
    file.write(reinterpret_cast<const char*>(DatasetMatrix_data), rows * cols * sizeof(double));

}

bool Dataset::has_all_targets(std::size_t row) const {

    for (const std::string& target : config_.targets) {
        if (const auto* numeric = std::get_if<NumericColumn>(&columns_.at(target))) {
            if (!numeric->values[row].has_value()) return false;
        } else if (const auto* categorical = std::get_if<CategoricalColumn>(&columns_.at(target))) {
            if (!categorical->values[row].has_value()) return false;
        }
    }
    return true;
}
ModelMatrix Dataset::encode(Layout encoding_type) {

    DatasetMatrix data;
    Targets targets;

    std::vector matrix_cols = columns_list;
    std::size_t matrix_rows = 0;

    // вычисление количества строк, где ВСЕ отклики имеют значение 
    for (std::size_t row = 0; row < rows; ++row) {
        if(has_all_targets(row)) ++matrix_rows;
    }

    // убираем отклики из вектора признаков и формируем массив словаря
    for (const std::string& target : config_.targets) {
        std::erase(matrix_cols, target);
    }


    switch (encoding_type)
    {
    case Layout::RowMajor: {
        std::size_t skipped_rows = 0;
        CategoricalFeatureInfo unique_values{};

        // формируем матрицу
        data = DatasetMatrix(matrix_rows, matrix_cols.size(), encoding_type, matrix_cols);
        for (std::size_t row = 0; row < rows; ++row) {
            // если отклик отсутствует, то скип строку
            if (!has_all_targets(row)) { ++skipped_rows; continue; }

            for (std::size_t column = 0; column < matrix_cols.size(); ++column) { 
                // row - skipped_rows --- подогнать индекс под массив длиной matrix_rows
                data(row - skipped_rows, column) = std::get<NumericColumn>(columns_[matrix_cols[column]]).values[row].value();
            }

        }

        for (const std::string& target : config_.targets) { // проход по каждому отклику
                skipped_rows = 0;
                if (auto* numeric = std::get_if<NumericColumn>(&columns_[target])) {
                    // если числовое, просто добавляем его
                    std::vector<double>& num_target = targets[target].emplace<NumericTarget>().values; 
                    num_target.resize(matrix_rows);

                    for (std::size_t row = 0; row < rows; ++row) {
                        if (!has_all_targets(row)) { ++skipped_rows; continue; }
                            num_target[row - skipped_rows] = numeric->values[row].value();
                    }
                } else if (auto* categorial = std::get_if<CategoricalColumn>(&columns_[target])) {
                    // если категориальное, то кодируем признаки в числа
                    CategorialTarget& cat_target = targets[target].emplace<CategorialTarget>();
                    cat_target.values.resize(matrix_rows);
                    unique_values = calculate_categorical_arr(target); // уникальные значения
                    std::size_t i = 0;
                    for (const std::string& value : unique_values.frequencies | std::views::keys) {
                        cat_target.categorical_values[value] = i++; // словарь значений {"STATUS": "D" : 0, "CL" : 1, "C" : 2}
                        // Таким образом, каждому категориальному признаку присваивается свой номер
                    }
                    for (std::size_t row = 0; row < rows; ++row) {
                        if (!has_all_targets(row)) { ++skipped_rows; continue; }
                        // в вектор значений откликов кладём уже закодированные строки
                        cat_target.values[row - skipped_rows] = cat_target.categorical_values[categorial->values[row].value()];
                    }
                }
            }
        break;
    }
    default:
        break;
    }
    return ModelMatrix(data, targets);
}

// kNNBase::kNNBase(DatasetMatrix& data, unsigned random_state=1) : 
//     train_data(data.get_rows(), data.get_cols()) {
//     std::vector<unsigned> num_rows(data.get_rows()); // номера строк
//     std::mt19937 engine(random_state);
//     for (int i = 1; i <= data.get_rows(); ++i) {
//         num_rows[i-1] = i;
//     }
//     // перемешивание массива номеров строк
//     std::shuffle(num_rows.begin(), num_rows.end(), engine);
    
//     for (int i = 0; i < data.get_rows(); ++i) {
//         train_data[i] = data[num_rows[i]];
//     }

// }
std::vector<std::string> Dataset::get_columns_list() const {
    return columns_list;
}

