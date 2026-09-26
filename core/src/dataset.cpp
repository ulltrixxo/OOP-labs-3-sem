#include <oop/dataset.hpp>
#include <fstream>
#include <format>
#include <string>
#include <type_traits>
#include <charconv>
#include <algorithm>
#include <variant>
#include <cmath>
#include <unordered_set>

std::optional<double> parse_double(std::string_view s) {
    double value = 0;
    auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    if (result.ec != std::errc{} || result.ptr != s.data() + s.size()) return std::nullopt;
    return value;
}
Dataset::Dataset(fs::path path) {
    std::vector<std::vector<std::string>> raw_data;
    std::ifstream file(path);

    if (!file) {
        return;
    }

    std::string data, header, value; 
    std::vector<std::string> values;

    // считывание первой строки датасета --- названия признаков.
    if(std::getline(file, data)) {
        std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
        while (std::getline(stream_data, value, ',')) {    
            if (value == "ID") continue; // Пропуск ID
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
            std::getline(stream_data, value, ',');
            if (i == 0) std::getline(stream_data, value, ','); // пропуск ID
            raw_data[i].push_back(value);
        }

    }
    rows = raw_data[0].size();

    // инициализация категорий датасета на основе сырых данных
    for (auto i = 0; i < columns_list.size(); ++i) {
        init_category(columns_list[i], raw_data[i]);
    }

    // преобразование сырых данных в готовых датасет
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns_list.size(); ++column) {
            if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[column]])) {
                auto& cat_arr = std::get<CategoricalColumn>(columns_[columns_list[column]]);
                if (raw_data[column][row] == "" || raw_data[column][row] == "NA" || raw_data[column][row] == "\"NA\"") {
                    cat_arr.values.push_back(std::nullopt);
                    ++missing_values[columns_list[column]];
                    continue;
                }
                cat_arr.values.push_back(raw_data[column][row]);
            } else {
                auto& num_arr = std::get<NumericColumn>(columns_[columns_list[column]]);
                if (raw_data[column][row] == "NA" || raw_data[column][row] == "\"NA\"") {
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
int Dataset::get_size() {
    return columns_.size();
}

// количество строк
int Dataset::get_rows() {
    return rows;
}

// инициализация категории на основе сырых данных
// если не получается перевести в число --- категориальный признак
// если получается перевести в число и уникальных значений > 10 --- числовой признак
// в противном случае --- категориальный признак
void Dataset::init_category(std::string category, std::vector<std::string> raw_data) {
    double num = 0;
    std::unordered_set<std::string> unique_values;
    for (std::size_t i = 0; i < raw_data.size(); ++i) {
        if (raw_data[i] != "NA" && raw_data[i] != "\"NA\"") {
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
            if (values[i] == "NA" || (values[i][0] == '\"' && values[i][values[i].length() - 1] == '\"')) {
                numeric_list.values.push_back(std::nullopt);
                ++missing_values[columns_list[i]];
                continue;
            }
            std::string_view values_ptr(values[i].data(), values[i].size());
            std::from_chars(values_ptr.data(), values_ptr.data() + values_ptr.size(), num);
            numeric_list.values.push_back(num);
        } else if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[i]])) {
            auto& categorial_list = std::get<CategoricalColumn>(columns_[columns_list[i]]);
            if (values[i] == "" || values[i] == "NA") {
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
void Dataset::analyze() {
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        info[columns_list[i]] = calculate(columns_list[i]);
    }
}

NumericFeatureInfo calculate_num_arr(std::span<double> array) {
    std::sort(array.begin(), array.end());
    NumericFeatureInfo result;
    double sum = 0;
    double squared_sum = 0;
    double min_st, max_st;
    int n = 0;

    for (auto i = 0; i < array.size(); ++i) {
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
    for (auto i = 0; i < array.size(); ++i) {
        squared_sum += (array[i] - result.mean) * (array[i] - result.mean);
    }
    result.variance = squared_sum / (array.size() - 1);
    double step = (result.max - result.min) / 10.;
    min_st = result.min;
    for (int i = 0; i < 10; ++i) {
        max_st = min_st + step;
        result.ranges[i].first = min_st;
        result.ranges[i].second = max_st;

        while (min_st <= array[n] && array[n] < max_st) {
                ++result.values[i];
                ++n;
        }
        min_st = max_st;
    }
    if (array[n] <= max_st) {
        ++result.values[9];
    }
    return result;
}

// высчитывание данных о каждом столбце датасета
FeatureInfo Dataset::calculate(std::string& category) {
    FeatureInfo statistic;
    if (std::holds_alternative<NumericColumn>(columns_[category])) {
        auto& numeric_arr = std::get<NumericColumn>(columns_[category]).values;
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
        numeric_stat.missing_count = missing_values[category];

    } else if (std::holds_alternative<CategoricalColumn>(columns_[category])) {
        statistic = CategoricalFeatureInfo{};
        auto& categorial_arr = std::get<CategoricalColumn>(columns_[category]).values;
        auto& categorial_stat = statistic.emplace<CategoricalFeatureInfo>();
        for (auto i = 0UL; i < categorial_arr.size(); ++i) {
            if (!categorial_arr[i].has_value()) {
                continue;
            }
            ++categorial_stat.frequencies[categorial_arr[i].value()];
        }
        categorial_stat.missing_count = missing_values[category];
    }

    return statistic;
}

// показать статистику столбца
int Dataset::show_stat(std::size_t n) {
    if (n < 0 || n > columns_list.size() - 1) return 1;
    
    std::cout << "\nПризнак " << n << ": " << columns_list[n];
    if (std::holds_alternative<NumericColumn>(columns_[columns_list[n]])) {
        auto& num_stat = std::get<NumericFeatureInfo>(info[columns_list[n]]);
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
        auto& cat_stat = std::get<CategoricalFeatureInfo>(info[columns_list[n]]);
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
                std::string(cat_stat.frequencies[categories[i]] / 10, '#'), 
                cat_stat.frequencies[categories[i]],
                static_cast<double>(cat_stat.frequencies[categories[i]]) / (rows - cat_stat.missing_count) * 100
            );
        }
    }
    std::cout << std::endl;

    return 0;
}

// показать общую сводку
void Dataset::show_info() {
    std::cout << std::format("{:>3} {:<14} {:<16} {:>8}\n", "#", "Признак", "Тип", "Пропуски");
    for (std::size_t i = 0UL; i < columns_list.size(); ++i) {
        const char* category = std::holds_alternative<NumericColumn>(columns_[columns_list[i]]) ? "Числовой" : "Категориальный";
        std::cout << std::format("{:>3} {:<14} {:<16} {:>8} ({:<.1f})%\n", i, columns_list[i], category, missing_values[columns_list[i]], static_cast<double>(missing_values[columns_list[i]])/rows*100);
    }
}
