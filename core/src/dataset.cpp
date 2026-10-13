#include <oop/dataset.hpp>
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


static Config config;
std::optional<double> parse_double(std::string_view s) {
    double value = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size()) return std::nullopt;
    return value;
}

bool Dataset::in_missing_values(const std::string& value) const {
    for (std::string missing_value : config.missing_marker) {
        if (value == missing_value) return true;
    }
    return false;
}
bool Dataset::in_missing_columns(const std::string& value) const {
    for (std::string missing_value : config.missing_columns) {
        if (value == missing_value) return true;
    }
    return false;
}
Dataset::Dataset(fs::path path) {
    std::vector<std::vector<std::string>> raw_data;
    std::ifstream file(path);

    config.delimiter = ',';
    config.missing_marker.push_back("NA");
    config.missing_marker.push_back("\"NA\"");
    config.missing_columns.push_back("ID");

    if (!file) {
        return;
    }

    std::string data, header, value; 
    std::vector<std::string> values;

    // считывание первой строки датасета --- названия признаков.
    if(std::getline(file, data)) {
        std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
        while (std::getline(stream_data, value, ',')) {    
            if (in_missing_columns(value)) continue; // Пропуск ID
            columns_list.push_back(value);
            missing_values[value] = 0;
        }
    }
    data.clear();
    raw_data.resize(columns_list.size());

    // заполнение сырыми данными
    while(std::getline(file, data)) {
        std::istringstream stream_data(data);
        for (auto i = 0UL; i < columns_list.size(); ++i) {
            std::getline(stream_data, value, config.delimiter);
            if (i == 0) std::getline(stream_data, value, config.delimiter); // пропуск ID
            raw_data[i].push_back(value);
        }

    }
    rows = raw_data[0].size();

    // инициализация категорий датасета на основе сырых данных
    for (std::size_t i = 0; i < columns_list.size(); ++i) {
        init_category(columns_list[i], raw_data[i]);
    }

    // преобразование сырых данных в готовых датасет
    for (int row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns_list.size(); ++column) {
            if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[column]])) {
                auto& cat_arr = std::get<CategoricalColumn>(columns_[columns_list[column]]);
                if (raw_data[column][row] == "" || in_missing_values(raw_data[column][row])) {
                    cat_arr.values.push_back(std::nullopt);
                    ++missing_values[columns_list[column]];
                    continue;
                }
                cat_arr.values.push_back(raw_data[column][row]);
            } else {
                auto& num_arr = std::get<NumericColumn>(columns_[columns_list[column]]);
                if (in_missing_values(raw_data[column][row])) {
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

// количество столбцов
int Dataset::get_size() const {
    return columns_.size();
}

// количество строк
int Dataset::get_rows() const {
    return rows;
}

// инициализация категории на основе сырых данных
// если не получается перевести в число --- категориальный признак
// если получается перевести в число и уникальных значений > 10 --- числовой признак
// в противном случае --- категориальный признак
void Dataset::init_category(const std::string& category, const std::vector<std::string>& raw_data) {
    double num = 0;
    std::unordered_set<std::string> unique_values;
    for (std::size_t i = 0; i < raw_data.size(); ++i) {
        if (!in_missing_values(raw_data[i])) {
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
            if (in_missing_values(values[i])) {
                numeric_list.values.push_back(std::nullopt);
                ++missing_values[columns_list[i]];
                continue;
            }
            std::string_view values_ptr(values[i].data(), values[i].size());
            std::from_chars(values_ptr.data(), values_ptr.data() + values_ptr.size(), num);
            numeric_list.values.push_back(num);
        } else if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[i]])) {
            auto& categorial_list = std::get<CategoricalColumn>(columns_[columns_list[i]]);
            if (values[i] == "" || in_missing_values(values[i])) {
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
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        info[columns_list[i]] = calculate(columns_list[i]);
    }
    return info;
}

NumericFeatureInfo calculate_num_arr(std::span<double> array) {
    std::sort(array.begin(), array.end());
    NumericFeatureInfo result;
    double sum = 0;
    double squared_sum = 0;
    double min_st, max_st;
    int n = 0;

    for (std::size_t i = 0; i < array.size(); ++i) {
        sum += array[i];
    }

    result.min = array[0];
    result.max = array[array.size() - 1];
    result.q05 = array[std::trunc(array.size() * 0.05) - 1];
    result.q25 = array[std::trunc(array.size() * 0.25) - 1];
    result.median = array[std::trunc(array.size() * 0.5) - 1];
    result.q75 = array[std::trunc(array.size() * 0.75) - 1];
    result.q95 = array[std::trunc(array.size() * 0.95) - 1];
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
        statistic = CategoricalFeatureInfo{};
        auto& categorial_arr = std::get<CategoricalColumn>(columns_.at(category)).values;
        auto& categorial_stat = statistic.emplace<CategoricalFeatureInfo>();
        for (auto i = 0UL; i < categorial_arr.size(); ++i) {
            if (!categorial_arr[i].has_value()) {
                continue;
            }
            ++categorial_stat.frequencies[categorial_arr[i].value()];
        }
        categorial_stat.missing_count = missing_values.at(category);
    }

    return statistic;
}

// показать статистику столбца
int Dataset::show_stat(const DatasetInfo& info, std::size_t n) const {
    if (n >= columns_list.size()) return 1;
    
    std::cout << "\nПризнак " << n << ": " << columns_list[n];
    if (std::holds_alternative<NumericColumn>(columns_.at(columns_list[n]))) {
        auto& num_stat = std::get<NumericFeatureInfo>(info.at(columns_list[n]));
        std::cout << " (числовой)" << std::endl;

        std::cout << "Пропусков: " << num_stat.missing_count << std::endl;
        std::cout << std::format("min: {:<.3f} max: {:<.3f}\n", num_stat.min, num_stat.max);
        std::cout << std::format("mean: {:<.3f} var: {:<.3f}\n", num_stat.mean, num_stat.variance) << std::endl;
        
        std::cout << "q05:" << num_stat.q05 << std::endl;
        std::cout << "q25:" << num_stat.q25 << std::endl;
        std::cout << "q50:" << num_stat.median << std::endl;
        std::cout << "q75:" << num_stat.q75 << std::endl;
        std::cout << "q95:" << num_stat.q95 << "\n" << std::endl;

        for (int i = 0; i < 10; ++i) {
            std::cout << std::format("[{:<.1f}, {:<.1f}) {:<20} {:<5}\n", num_stat.ranges[i].first, num_stat.ranges[i].second, std::string(num_stat.values[i] / 10, '#'), num_stat.values[i]);
        }

    } else {
        auto& cat_stat = std::get<CategoricalFeatureInfo>(info.at(columns_list[n]));
        std::vector<std::string> categories;
        std::cout << " (категориальный)" << std::endl;

        std::cout << "Пропусков: " << cat_stat.missing_count << std::endl;

        std::cout << "Категории:\n";
        int i = 1;
        for (const auto& [key, value] : cat_stat.frequencies) {
            categories.push_back(key);
            std::cout << i++ << ". " << key << "\n";
        }
        std::cout << "Гистограмма по категориям:" << std::endl;

        for (auto i = 0UL; i < cat_stat.frequencies.size(); ++i) {
            std::cout << std::format("{:<10} {:<20} {:<5} ({:<.1f}%)\n", 
                categories[i], 
                std::string(cat_stat.frequencies.at(categories[i]) / 10, '#'), 
                cat_stat.frequencies.at(categories[i]),
                static_cast<double>(cat_stat.frequencies.at(categories[i])) / (rows - cat_stat.missing_count) * 100
            );
        }
    }
    std::cout << std::endl;

    return 0;
}

// показать общую сводку
void Dataset::show_info() const {
    std::cout << std::format("{:>3} {:<14} {:<16} {:>8}\n", "#", "Признак", "Тип", "Пропуски");
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        const char* category = std::holds_alternative<NumericColumn>(columns_.at(columns_list[i])) ? "Числовой" : "Категориальный";
        std::cout << std::format("{:>3} {:<14} {:<16} {:>8} ({:<.1f})%\n", i, columns_list[i], category, missing_values.at(columns_list[i]), static_cast<double>(missing_values.at(columns_list[i]))/rows*100);
        }
}

void Dataset::impute(DatasetInfo& info) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (const std::string& column : columns_list) {
            if (column == "N_Days" || column == "Status") continue;
            if (std::holds_alternative<NumericColumn>(columns_[column])) {
                auto& num_arr = std::get<NumericColumn>(columns_[column]).values;
                auto& num_stat = std::get<NumericFeatureInfo>(info[column]);
                for (std::optional<double>& x : num_arr) {
                    if (!x.has_value()) {
                        x = num_stat.median;
                    }
                }
            } else if (std::holds_alternative<CategoricalColumn>(columns_[column])) {
                auto& cat_arr = std::get<CategoricalColumn>(columns_[column]).values;
                for (std::optional<std::string>& x : cat_arr) {
                    if (!x.has_value()) {
                        x = "__MISSING__";
                    }
                }
            }
        }
    }
}

void Dataset::scale_numeric_features(DatasetInfo& info) {
    for (const std::string& column : columns_list) {
        if (std::holds_alternative<NumericColumn>(columns_[column])) {
            auto& num_arr = std::get<NumericColumn>(columns_[column]).values;
            auto& num_stat = std::get<NumericFeatureInfo>(info[column]);
            for (std::optional<double>& x : num_arr) {
                x = (x.value() - num_stat.mean) / sqrt(num_stat.variance);
            }
        } 
    }
}

void Dataset::OneHot(DatasetInfo& info) {
    std::vector<std::string> new_column_list;
    std::string onehot_header;
    for (const std::string& column : columns_list) {
        if (std::holds_alternative<CategoricalColumn>(columns_[column])) {
            std::vector<std::string> onehot_headers;
            std::vector<std::string> onehot_values;
            auto& cat_arr = std::get<CategoricalColumn>(columns_[column]).values;
            auto& cat_stat = std::get<CategoricalFeatureInfo>(info[column]);
            for (const auto& unique_value : cat_stat.frequencies) {
                onehot_header = column + "=" + unique_value.first;
                columns_[onehot_header].emplace<NumericColumn>().values.resize(rows);
                onehot_headers.push_back(onehot_header);
                onehot_values.push_back(unique_value.first);
            }
            std::sort(onehot_headers.begin(), onehot_headers.end());
            std::sort(onehot_values.begin(), onehot_values.end());
            if (cat_stat.missing_count > 0){
                onehot_headers.push_back(column + "=__MISSING__");
                onehot_values.push_back("__MISSING__");
                columns_[column + "=__MISSING__"].emplace<NumericColumn>().values.resize(rows);
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

DatasetMatrix::DatasetMatrix(std::size_t rows, std::size_t cols, Layout layout,
const std::vector<std::string>& feature_names) : feature_names(feature_names), rows_(rows), cols_(cols), layout_(layout){
    data_.resize(rows*cols);
}
double& DatasetMatrix::operator()(std::size_t row, std::size_t column) {
    return data_[column + cols_*row];
}
const double& DatasetMatrix::operator()(std::size_t row, std::size_t column) const {
    return data_[column + cols_*row];
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
DatasetMatrix Dataset::encode(Layout encoding_type) {
    DatasetMatrix result;
    switch (encoding_type)
    {
    case Layout::RowMajor:
        result = DatasetMatrix(rows, columns_list.size(), encoding_type, columns_list);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns_list.size(); ++column) {
                result(row, column) = std::get<NumericColumn>(columns_[columns_list[column]]).values[row].value();
            }
        }
        break;
    
    default:
        break;
    }
    return result;
}

const double* DatasetMatrix::get_data() const {
    return data_.data();
}
std::string_view DatasetMatrix::get_feature_name(std::size_t i) {
    return feature_names[i];
}
void save(DatasetMatrix& data) {
    std::ofstream file("data/cirrhosis.bin", std::ios::binary);
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

DatasetMatrix DatasetMatrix::load(
    const std::filesystem::path& path
) {
    std::ifstream file(path, std::ios::binary);
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