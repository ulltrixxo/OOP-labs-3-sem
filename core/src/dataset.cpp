#include <oop/dataset.hpp>
#include <fstream>
#include <format>
#include <string>
#include <type_traits>
#include <charconv>


Dataset::Dataset(fs::path path) {
    std::ifstream file(path);

    if (!file) {
        return;
    }

    std::string data, header, value; 
    std::vector<std::string> values;
    if(std::getline(file, data)) { // считывание первой строки датасета --- названия признаков.
        std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
        while (std::getline(stream_data, value, ',')) {    
            if (value == "ID") continue;    
            columns_list.push_back(value);
        }
    }
    data.clear();
    missing.resize(columns_list.size());

    double num = 0.0;

    // инициализация категорий столбцов датасета
    if(std::getline(file, data)) {
        std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
        for (auto i = 0UL; i < columns_list.size(); ++i) {
            std::getline(stream_data, value, ',');
            if (i == 0) std::getline(stream_data, value, ',');
            values.push_back(value);
            std::string_view value_ptr(value.data(), value.size());

            if (value[0] == '\"' && value[value.length() - 1] == '\"'){
                value.erase(0, 1);
                value.erase(value.length()-1, 1);
            }

            auto [ptr, ec] = std::from_chars(value_ptr.data(), value_ptr.data() + value_ptr.size(), num);
            header = columns_list[i];
            if (ec == std::errc()) {
                this->init_category(header, 'N');
            } else if (value == "NA") {
                this->init_category(header, 'N');
            } else {
                this->init_category(header, 'C');
            }
        }
    }


    // заполнение датасета данными
    this->add_row(values);
    values.clear();
    data.clear();
    while (std::getline(file, data)) {
        std::istringstream stream_data(data); // превращение data в поток, нужный для getline()
        for (auto i = 0UL; i < columns_list.size(); ++i) {
            std::getline(stream_data, value, ',');
            if (i == 0) std::getline(stream_data, value, ',');
            values.push_back(value);
        }
        this->add_row(values);
        values.clear();
        data.clear();
    }

}
void Dataset::init_category(std::string str, char category) {
    if (category == 'N') {
        columns_[str] = NumericColumn{};
    } else {
        columns_[str] = CategoricalColumn{};
    }
}

void Dataset::add_row(std::vector<std::string>& values) {
    double num = 0.0;
    for (auto i = 0UL; i < columns_list.size(); ++i) {
        if (std::holds_alternative<NumericColumn>(columns_[columns_list[i]])){
            auto& numeric_list = std::get<NumericColumn>(columns_[columns_list[i]]);
            if (values[i] == "NA" || (values[i][0] == '\"' && values[i][values[i].length() - 1] == '\"')) {
                numeric_list.values.push_back(std::nullopt);
                ++missing[i];
                continue;
            }
            std::string_view values_ptr(values[i].data(), values[i].size());
            auto [ptr, ec] = std::from_chars(values_ptr.data(), values_ptr.data() + values_ptr.size(), num);
            numeric_list.values.push_back(num);
        } else if (std::holds_alternative<CategoricalColumn>(columns_[columns_list[i]])) {
            auto& categorial_list = std::get<CategoricalColumn>(columns_[columns_list[i]]);
            if (values[i] == "") ++missing[i];
            categorial_list.values.push_back(values[i]);
        }
    }
    ++rows;
}

void Dataset::show_info() {
    // std::format("{:>}");
}
