// Лабораторная работа 1. Загрузка, анализ и предобработка табличных данных.
#include "oop/dataset.hpp"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <charconv>
#include <iostream>

int main() {

    Dataset dataset("data/cirrhosis.csv");

    dataset.analyze();

    dataset.show_info();

    int n;
    std::string line;
    bool status;
    do {
        std::cout << "Введите номер признака, для которого нужно вывести статистику (для выхода введите q): ";
        status = std::getline(std::cin, line) && line != "q";
        const auto [ptr, ec] = std::from_chars(line.data(), line.data() + line.size(), n);
        if (ec != std::errc{} || ptr != line.data() + line.size()) {
            std::cout << "Введено некорректное значение, попробуйте ещё раз: " << std::endl;
            continue;
        }
        if (dataset.show_stat(n)) {
            std::cout << "Число вне диапазона. Попробуйте ещё раз: " << std::endl;
        }
    } while (status);

    return 0;
}
