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
#include <iostream>


int show_stat(const DatasetInfo& info, std::size_t n) {
    if (n >= info.columns.size()) return 1;
    
    std::cout << "\nПризнак " << n << ": " << info.columns[n];
    const FeatureInfo& feature = info.features.at(info.columns[n]);
    if (std::holds_alternative<NumericFeatureInfo>(feature)) {
        auto& num_stat = std::get<NumericFeatureInfo>(feature);
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
        auto& cat_stat = std::get<CategoricalFeatureInfo>(feature);
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
                static_cast<double>(cat_stat.frequencies.at(categories[i])) / (info.rows - cat_stat.missing_count) * 100
            );
        }
    }
    std::cout << std::endl;

    return 0;
}

// показать общую сводку
void show_info(const DatasetInfo& info) {
    std::cout << std::format("{:>3} {:<14} {:<16} {:>8}\n", "#", "Признак", "Тип", "Пропуски");
    for (std::size_t i = 0UL; i < info.columns.size(); ++i) {
        const FeatureInfo& feature = info.features.at(info.columns[i]);
        const char* category = std::holds_alternative<NumericFeatureInfo>(feature) ? "Числовой" : "Категориальный";
        int missings = std::holds_alternative<NumericFeatureInfo>(feature) ? 
        std::get<NumericFeatureInfo>(feature).missing_count : std::get<CategoricalFeatureInfo>(feature).missing_count;

        double stat_res = std::holds_alternative<NumericFeatureInfo>(feature) ? 
        static_cast<double>(missings)/info.rows*100:
        static_cast<double>(missings)/info.rows*100;

        std::cout << std::format(
            "{:>3} {:<14} {:<16} {:>8} ({:<.1f})%\n",
            i, 
            info.columns[i], 
            category, 
            missings, 
            stat_res
        );
    }
}