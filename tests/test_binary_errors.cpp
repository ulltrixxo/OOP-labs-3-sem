// Бинарный формат: реакция на некорректный файл.
//
// Условие работы 1: ошибки ввода-вывода обрабатываются осознанно, сообщением,
// а не аварийным завершением. DatasetMatrix::load должна бросать исключение,
// если файла нет или он обрезан.
//
// Тест вынесен в отдельную программу: если load не проверяет поток, она читает
// неинициализированные размеры и может запросить гигантский объём памяти —
// программа аварийно остановится, и это не должно мешать остальным тестам.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <exception>

int main()
{
    const auto dir = fixtures::temp_dir("binary_errors");

    // Файла нет.
    CHECK_THROWS_AS(DatasetMatrix::load(dir / "missing.bin"), std::exception);

    // Файл обрезан посреди заголовка: 5 байт вместо размеров матрицы.
    const auto truncated = fixtures::write_file(dir / "truncated.bin", "12345");
    CHECK_THROWS_AS(DatasetMatrix::load(truncated), std::exception);

    return testing::summary();
}
