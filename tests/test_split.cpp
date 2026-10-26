// Разбиение на обучающую и тестовую части (работа 2).
//
// Проверяется на искусственных данных, где по каждой строке видно, откуда она
// взялась: первый признак — номер исходной строки, второй — номер + 100,
// числовой отклик — номер × 10, категориальный — номер % 3. Если строки
// перепутаются или ответы съедут относительно признаков, это сразу видно.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"
#include "mlcore/matrix.hpp"

#include <cstddef>
#include <format>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace
{

const std::map<std::string, int> kCodes = {{"a", 0}, {"b", 1}, {"c", 2}};

ModelMatrix make_data(std::size_t rows)
{
    DatasetMatrix matrix(rows, 2, Layout::RowMajor, {"id", "id_plus_100"});
    NumericTarget numeric;
    CategorialTarget categorical;
    categorical.categorical_values = kCodes;

    for (std::size_t r = 0; r < rows; ++r)
    {
        matrix(r, 0) = static_cast<double>(r);
        matrix(r, 1) = static_cast<double>(r) + 100;
        numeric.values.push_back(static_cast<double>(r) * 10);
        categorical.values.push_back(static_cast<double>(r % 3));
    }

    Targets targets;
    targets["num"] = numeric;
    targets["cat"] = categorical;
    return ModelMatrix(matrix, targets);
}

// Номера исходных строк в порядке, в каком они лежат в части.
std::vector<double> ids_of(const ModelMatrix& part)
{
    std::vector<double> ids;
    for (std::size_t r = 0; r < part.get_data().get_rows(); ++r)
    {
        ids.push_back(part.get_data()(r, 0));
    }
    return ids;
}

// --- Размеры частей ---------------------------------------------------------------

// Обучающая часть — round(rows × proportion) строк, тестовая — остальные.
void test_sizes()
{
    const ModelMatrix source = make_data(10);

    const Split seventy = source.split(0.7, 42);
    CHECK(seventy.train.get_data().get_rows() == 7);
    CHECK(seventy.test.get_data().get_rows() == 3);

    // 10 × 0.25 = 2.5, std::round округляет половину от нуля → 3.
    const Split quarter = source.split(0.25, 42);
    CHECK(quarter.train.get_data().get_rows() == 3);
    CHECK(quarter.test.get_data().get_rows() == 7);
}

// --- Части не пересекаются и покрывают всю выборку --------------------------------

void test_partition()
{
    const Split split = make_data(10).split(0.7, 42);

    std::multiset<double> all;
    for (const double id : ids_of(split.train)) all.insert(id);
    for (const double id : ids_of(split.test))  all.insert(id);

    CHECK(all.size() == 10);                                 // ни одной строки не потеряно
    CHECK(std::set<double>(all.begin(), all.end()).size() == 10);   // и ни одна не попала дважды
    for (int r = 0; r < 10; ++r)
    {
        CHECK(all.count(r) == 1);
    }
}

// --- Признаки и ответы остаются вместе --------------------------------------------

// Строка части и её ответы относятся к одной исходной строке — для обоих
// типов отклика. Таблица кодов категориального отклика есть в обеих частях.
void check_aligned(const ModelMatrix& part)
{
    const DatasetMatrix& data = part.get_data();
    const auto* numeric = std::get_if<NumericTarget>(&part.get_targets().at("num"));
    const auto* categorical = std::get_if<CategorialTarget>(&part.get_targets().at("cat"));
    CHECK(numeric != nullptr && categorical != nullptr);
    if (!numeric || !categorical) return;

    CHECK(numeric->values.size() == data.get_rows());
    CHECK(categorical->values.size() == data.get_rows());
    CHECK(categorical->categorical_values == kCodes);
    if (numeric->values.size() != data.get_rows() || categorical->values.size() != data.get_rows()) return;

    for (std::size_t r = 0; r < data.get_rows(); ++r)
    {
        const double id = data(r, 0);
        CHECK(data(r, 1) == id + 100);
        CHECK(numeric->values[r] == id * 10);
        CHECK(categorical->values[r] == static_cast<double>(static_cast<int>(id) % 3));
    }
}

void test_alignment()
{
    const Split split = make_data(10).split(0.7, 42);
    check_aligned(split.train);
    check_aligned(split.test);
}

// --- Воспроизводимость по random_state --------------------------------------------

void test_random_state()
{
    const ModelMatrix source = make_data(10);

    // Одинаковый random_state — одинаковое разбиение, вплоть до порядка строк.
    const Split first = source.split(0.7, 42);
    const Split again = source.split(0.7, 42);
    CHECK(ids_of(first.train) == ids_of(again.train));
    CHECK(ids_of(first.test) == ids_of(again.test));

    // Другой random_state — другое разбиение. Для 10 строк совпадение порядка
    // при разных начальных значениях практически невозможно, а для этой
    // стандартной библиотеки результат фиксирован.
    const Split other = source.split(0.7, 7);
    CHECK(ids_of(first.train) != ids_of(other.train));

    // Строки действительно перемешаны, а не взяты по порядку: в cirrhosis.csv
    // пациенты вне испытания стоят в конце файла, и разбиение «первые 70%»
    // отправило бы их всех в тест.
    CHECK((ids_of(first.train) != std::vector<double>{0, 1, 2, 3, 4, 5, 6}));
}

// --- Ошибки -------------------------------------------------------------------------

void test_errors()
{
    const ModelMatrix source = make_data(10);

    // Доля вне отрезка [0; 1].
    CHECK_THROWS_AS(source.split(-0.1, 42), std::runtime_error);
    CHECK_THROWS_AS(source.split(1.5, 42), std::runtime_error);

    // Инвариант ModelMatrix: ответов столько же, сколько строк матрицы.
    DatasetMatrix matrix(3, 1, Layout::RowMajor, {"x"});
    Targets targets;
    targets["num"] = NumericTarget{{1.0, 2.0}};   // 2 ответа на 3 строки
    CHECK_THROWS_AS(ModelMatrix(matrix, targets), std::runtime_error);
}

// --- Разбиение настоящей предобработки ---------------------------------------------

// Весь конвейер: CSV → предобработка → encode → split. Части вместе дают все
// строки, у каждой части ответов столько же, сколько строк.
void test_pipeline()
{
    std::string text = "ID,num,color,status\n";
    for (int r = 0; r < 30; ++r)
    {
        const char* status = r % 3 == 0 ? "C" : (r % 3 == 1 ? "D" : "CL");
        text += std::format("{},{},{},{}\n", r, r * 1.5, r % 2 == 0 ? "red" : "blue", status);
    }
    const auto path = fixtures::write_file(fixtures::temp_dir("split_pipeline") / "data.csv", text);

    Config config = fixtures::default_config();
    config.targets = {"status"};
    Dataset dataset(path, config);
    DatasetInfo info = dataset.analyze();
    dataset.impute(info);
    dataset.scale_numeric_features(info);
    dataset.OneHot(info);
    const ModelMatrix encoded = dataset.encode();

    const Split split = encoded.split(0.7, 42);
    const auto rows_of = [](const ModelMatrix& part) { return part.get_data().get_rows(); };
    const auto targets_of = [](const ModelMatrix& part) {
        return std::get<CategorialTarget>(part.get_targets().at("status")).values.size();
    };

    CHECK(rows_of(split.train) + rows_of(split.test) == 30);
    CHECK(rows_of(split.train) == 21);
    CHECK(targets_of(split.train) == rows_of(split.train));
    CHECK(targets_of(split.test) == rows_of(split.test));
}

} // namespace

int main()
{
    test_sizes();
    test_partition();
    test_alignment();
    test_random_state();
    test_errors();
    test_pipeline();

    return testing::summary();
}
