#pragma once

// Общие заготовки для тестов: конфигурация загрузчика и временные файлы.
// Тесты сами создают маленькие CSV во временном каталоге, поэтому не зависят
// от data/cirrhosis.csv и не трогают его.

#include "mlcore/dataset.hpp"

#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>
#include <string_view>

namespace fixtures
{

// Та же конфигурация, что в приложении: пропуск записывается как NA или "NA",
// столбец ID не является признаком. Отклика нет: в тестовых CSV все столбцы —
// признаки. Тесты, которым нужен отклик, задают targets сами.
inline Config default_config()
{
    return Config{
        .delimiter = ',',
        .missing_values = {"NA", "\"NA\""},
        .unnecessary_columns = {"ID"},
        .impute_missing_value = "__MISSING__",
        .targets = {},
    };
}

// Пустой каталог для временных файлов одного теста. Если остался от прошлого
// запуска — очищается, чтобы старые файлы не влияли на результат.
inline std::filesystem::path temp_dir(std::string_view test_name)
{
    const auto dir = std::filesystem::temp_directory_path()
                   / ("oop_tests_" + std::string(test_name));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// Записывает содержимое в файл и возвращает путь к нему.
inline std::filesystem::path write_file(const std::filesystem::path& path,
                                        std::string_view content)
{
    std::ofstream file(path, std::ios::binary);
    file << content;
    return path;
}

// Собирает строки в текст CSV, по строке на элемент, с переводом строки в конце.
inline std::string csv(std::initializer_list<std::string_view> lines)
{
    std::string text;
    for (const auto line : lines)
    {
        text += line;
        text += '\n';
    }
    return text;
}

} // namespace fixtures
