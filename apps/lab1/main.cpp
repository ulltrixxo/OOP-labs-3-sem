// Лабораторная работа 1. Загрузка, анализ и предобработка табличных данных.
#include "oop/dataset.hpp"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <charconv>
#include <iostream>
#include <cmath>
#include <format>

int main() {

    Dataset dataset("data/cirrhosis.csv");

    DatasetInfo info = dataset.analyze();

    dataset.show_info();

    int n;
    char save_status = 0;
    std::string line;
    while (line != "q") {
        std::cout << "Введите номер признака, для которого нужно вывести статистику (для выхода введите q): ";
        std::getline(std::cin, line);
        if (line == "q") break;
        const auto [ptr, ec] = std::from_chars(line.data(), line.data() + line.size(), n);
        if (ec != std::errc{} || ptr != line.data() + line.size()) {
            std::cout << "Введено некорректное значение, попробуйте ещё раз: " << std::endl;
            continue;
        }
        if (dataset.show_stat(info, n)) {
            std::cout << "Число вне диапазона. Попробуйте ещё раз: " << std::endl;
        }
    };

    std::cout << "Сохранить файл? (y, n): ";
    std::cin >> save_status;
    while (save_status != 'y' && save_status != 'n') {
        std::cout << "Некорректное значение, попробуйте ещё раз: ";
        std::cin >> save_status;
    }

    if (save_status == 'n') return 0;

    dataset.impute(info);
    dataset.scale_numeric_features(info);
    dataset.OneHot(info);
    DatasetMatrix matrix = dataset.encode();
    save(matrix);
    DatasetMatrix loaded_matrix = DatasetMatrix::load("data/cirrhosis.bin");

    const auto csv_size = std::filesystem::file_size("data/cirrhosis.csv"); 
    const auto bin_size = std::filesystem::file_size("data/cirrhosis.bin");
    
    std::cout << std::format("CSV: {:>8} байт\n", csv_size);
    std::cout << std::format("Binary: {:>8} байт\n", bin_size);
    return 0;
}
