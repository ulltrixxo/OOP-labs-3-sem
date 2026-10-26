// Загрузка CSV: распознавание пропусков, определение типа признака,
// независимость от порядка столбцов и реакция на некорректные файлы.
//
// Обязательные проверки из условия работы 1, которые покрывает этот файл:
//   - разбор строки с пропуском;
//   - определение типа признака;
//   - реакция на несогласованный CSV.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <exception>
#include <format>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace
{

const NumericFeatureInfo* as_numeric(const DatasetInfo& info, const std::string& name)
{
    return std::get_if<NumericFeatureInfo>(&info.features.at(name));
}

const CategoricalFeatureInfo* as_categorical(const DatasetInfo& info, const std::string& name)
{
    return std::get_if<CategoricalFeatureInfo>(&info.features.at(name));
}

DatasetInfo load_and_analyze(const fs::path& path)
{
    const Dataset dataset(path, fixtures::default_config());
    return dataset.analyze();
}

// Текст исключения, брошенного при загрузке, или пустая строка, если загрузка
// прошла без ошибки. Нужен, чтобы проверить содержание сообщения, а не только
// факт исключения.
std::string load_error(const fs::path& path)
{
    try
    {
        const Dataset dataset(path, fixtures::default_config());
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    return "";
}

bool contains(const std::string& text, const std::string& part)
{
    return text.find(part) != std::string::npos;
}

// --- Разбор строк с пропусками ---------------------------------------------

// NA и "NA" (в кавычках, как в столбце Stage исходного файла) должны
// становиться пропусками и в числовом, и в категориальном признаке. Пропуски
// не участвуют в статистике: min и max считаются только по настоящим числам.
//
// Строк 24, а не меньше: при выборке короче 20 значений analyze() сейчас
// падает на расчёте q05 (см. test_statistics_small), а этот тест не о квантилях.
void test_missing_values()
{
    const auto dir = fixtures::temp_dir("loader_missing");

    std::string text = "ID,num,cat\n";
    for (int i = 1; i <= 24; ++i)
    {
        std::string num = std::to_string(i);
        if (i == 3)  num = "NA";
        if (i == 7)  num = "\"NA\"";

        std::string cat = (i % 2 == 0) ? "A" : "B";
        if (i == 5)  cat = "NA";
        if (i == 9)  cat = "\"NA\"";

        text += std::format("{},{},{}\n", i, num, cat);
    }
    const auto info = load_and_analyze(fixtures::write_file(dir / "data.csv", text));

    CHECK(info.rows == 24);
    CHECK((info.columns == std::vector<std::string>{"num", "cat"}));   // ID пропущен

    const auto* num = as_numeric(info, "num");
    CHECK(num != nullptr);
    if (num)
    {
        CHECK(num->missing_count == 2);
        CHECK(num->min == 1.0);
        CHECK(num->max == 24.0);
    }

    const auto* cat = as_categorical(info, "cat");
    CHECK(cat != nullptr);
    if (cat)
    {
        CHECK(cat->missing_count == 2);
        CHECK(cat->frequencies.size() == 2);                  // только A и B
        CHECK(!cat->frequencies.contains("NA"));
        CHECK(!cat->frequencies.contains("\"NA\""));
        CHECK(cat->frequencies.at("A") + cat->frequencies.at("B") == 22);
    }
}

// --- Определение типа признака ---------------------------------------------

// Правило из dataset.cpp: не разбирается как число — категориальный; число,
// но различных значений не больше 10 — категориальный; иначе числовой.
void test_type_detection()
{
    const auto dir = fixtures::temp_dir("loader_types");

    std::string text = "ID,text,few,ten,eleven\n";
    for (int i = 0; i < 22; ++i)
    {
        text += std::format("{},x{},{},{},{}\n",
                            i,
                            i % 3,          // текст
                            1 + i % 4,      // 4 различных числа, как Stage
                            1 + i % 10,     // ровно 10 — граница правила
                            1 + i % 11);    // 11 — уже числовой
    }
    const auto info = load_and_analyze(fixtures::write_file(dir / "data.csv", text));

    CHECK(as_categorical(info, "text") != nullptr);
    CHECK(as_categorical(info, "few") != nullptr);
    CHECK(as_categorical(info, "ten") != nullptr);
    CHECK(as_numeric(info, "eleven") != nullptr);
}

// По правилу «не разбирается как число — категориальный» результат не должен
// зависеть от того, где в столбце стоит нечисловое значение. Здесь оно стоит
// после 24 различных чисел.
void test_type_detection_late_text()
{
    const auto dir = fixtures::temp_dir("loader_late_text");

    std::string text = "ID,late_word,late_suffix\n";
    for (int i = 1; i <= 24; ++i)
    {
        text += std::format("{},{},{}\n", i, i, i);
    }
    text += "25,abc,12abc\n";
    const auto info = load_and_analyze(fixtures::write_file(dir / "data.csv", text));

    CHECK(as_categorical(info, "late_word") != nullptr);
    CHECK(as_categorical(info, "late_suffix") != nullptr);
}

// --- Порядок столбцов --------------------------------------------------------

// Условие: изменение порядка столбцов не ломает программу. Те же данные со
// столбцами в другом порядке и ID в середине дают ту же статистику по именам.
void test_column_order()
{
    const auto dir = fixtures::temp_dir("loader_order");

    std::string original = "ID,a,b,c\n";
    std::string permuted = "c,ID,b,a\n";
    for (int i = 1; i <= 25; ++i)
    {
        const std::string a = std::to_string(i * 3);
        const std::string b = (i % 3 == 0) ? "NA" : (i % 2 == 0 ? "yes" : "no");
        const std::string c = (i % 5 == 0) ? "NA" : std::to_string(100 + i);

        original += std::format("{},{},{},{}\n", i, a, b, c);
        permuted += std::format("{},{},{},{}\n", c, i, b, a);
    }
    const auto first  = load_and_analyze(fixtures::write_file(dir / "original.csv", original));
    const auto second = load_and_analyze(fixtures::write_file(dir / "permuted.csv", permuted));

    CHECK((first.columns  == std::vector<std::string>{"a", "b", "c"}));
    CHECK((second.columns == std::vector<std::string>{"c", "b", "a"}));
    CHECK(first.rows == second.rows);

    for (const std::string name : {"a", "c"})
    {
        const auto* x = as_numeric(first, name);
        const auto* y = as_numeric(second, name);
        CHECK(x != nullptr && y != nullptr);
        if (x && y)
        {
            CHECK(x->missing_count == y->missing_count);
            CHECK(x->min == y->min);
            CHECK(x->max == y->max);
            CHECK(x->mean == y->mean);
        }
    }

    const auto* x = as_categorical(first, "b");
    const auto* y = as_categorical(second, "b");
    CHECK(x != nullptr && y != nullptr);
    if (x && y)
    {
        CHECK(x->missing_count == y->missing_count);
        CHECK(x->frequencies == y->frequencies);
    }
}

// --- Несколько пропускаемых столбцов -------------------------------------------

// Конфигурация может исключать несколько столбцов, например ID и N_Days, чтобы
// сравнить модели с признаком и без него. Пропускаемые столбцы стоят не подряд:
// загрузчик должен сдвигать соответствие «поле файла — столбец датасета» каждый
// раз, а не один.
void test_several_skipped_columns()
{
    const auto dir = fixtures::temp_dir("loader_skip_several");

    std::string text = "a,ID,b,days,c\n";
    for (int i = 1; i <= 25; ++i)
    {
        text += std::format("{},{},{},{},{}\n",
                            i,                       // a
                            1000 + i,                // ID — пропускается
                            i % 2 == 0 ? "x" : "y",  // b
                            5000 + i,                // days — пропускается
                            100 + i);                // c
    }
    const auto path = fixtures::write_file(dir / "data.csv", text);

    Config config = fixtures::default_config();
    config.unnecessary_columns = {"ID", "days"};
    const Dataset dataset(path, config);
    const DatasetInfo info = dataset.analyze();

    CHECK((info.columns == std::vector<std::string>{"a", "b", "c"}));

    // Если бы пропуск сдвинул столбцы, в a, b или c оказались бы значения ID или days.
    const auto* a = as_numeric(info, "a");
    const auto* c = as_numeric(info, "c");
    CHECK(a != nullptr && c != nullptr);
    if (a && c)
    {
        CHECK(a->min == 1.0 && a->max == 25.0);
        CHECK(c->min == 101.0 && c->max == 125.0);
    }
    const auto* b = as_categorical(info, "b");
    CHECK(b != nullptr);
    if (b)
    {
        CHECK(b->frequencies.size() == 2);
    }
}

// --- Некорректные файлы -------------------------------------------------------

void test_errors()
{
    const auto dir = fixtures::temp_dir("loader_errors");
    const std::string header = "ID,a,b";

    // Файла нет.
    CHECK_THROWS_AS(Dataset(dir / "missing.csv", fixtures::default_config()), std::runtime_error);

    // Пустой файл.
    const auto empty = fixtures::write_file(dir / "empty.csv", "");
    CHECK_THROWS_AS(Dataset(empty, fixtures::default_config()), std::runtime_error);

    // Первая строка пустая — заголовка нет.
    const auto blank = fixtures::write_file(dir / "blank.csv", fixtures::csv({"", header, "1,2,x"}));
    CHECK_THROWS_AS(Dataset(blank, fixtures::default_config()), std::runtime_error);

    // Только заголовок, строк данных нет.
    const auto header_only = fixtures::write_file(dir / "header_only.csv", fixtures::csv({header}));
    CHECK_THROWS_AS(Dataset(header_only, fixtures::default_config()), std::runtime_error);

    // В заголовке только пропускаемый столбец — признаков нет.
    const auto id_only = fixtures::write_file(dir / "id_only.csv", fixtures::csv({"ID", "1", "2"}));
    CHECK_THROWS_AS(Dataset(id_only, fixtures::default_config()), std::runtime_error);

    // Несогласованный CSV: в строке 5 файла не хватает поля. Сообщение должно
    // называть номер строки, считая заголовок первой строкой.
    const auto fewer = fixtures::write_file(dir / "fewer.csv",
        fixtures::csv({header, "1,2,x", "2,3,y", "3,4,z", "4,5", "5,6,w"}));
    CHECK_THROWS_AS(Dataset(fewer, fixtures::default_config()), std::runtime_error);
    CHECK(contains(load_error(fewer), "line 5"));

    // Лишнее поле в строке 3.
    const auto more = fixtures::write_file(dir / "more.csv",
        fixtures::csv({header, "1,2,x", "2,3,y,extra", "3,4,z"}));
    CHECK_THROWS_AS(Dataset(more, fixtures::default_config()), std::runtime_error);
    CHECK(contains(load_error(more), "line 3"));

    // Пустое последнее поле — это не ошибка формы строки: полей столько же,
    // сколько в заголовке. Загрузка не должна падать.
    const auto trailing = fixtures::write_file(dir / "trailing.csv",
        fixtures::csv({header, "1,2,x", "2,3,", "3,4,z"}));
    CHECK(load_error(trailing).empty());
}

} // namespace

int main()
{
    test_missing_values();
    test_type_detection();
    test_type_detection_late_text();
    test_column_order();
    test_several_skipped_columns();
    test_errors();

    return testing::summary();
}
