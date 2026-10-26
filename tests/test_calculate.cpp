// Статистика числового признака.
//
// Расчёт статистики — приватный метод Dataset, поэтому массивы попадают в него
// так же, как в программе: через CSV, загрузку и analyze().
//
// Две проверки:
//   - известный массив, ответ для которого посчитан вручную (обязательная
//     проверка условия «расчёт квантилей на известном массиве»);
//   - свойства, которые должны выполняться на любых данных, на случайном массиве.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <algorithm>
#include <cstddef>
#include <format>
#include <numeric>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace
{

// Возвращает count случайных чисел, равномерно распределённых в [min, max).
//
// seed делает результат воспроизводимым: одно и то же значение даёт один и тот
// же массив при каждом запуске. Без этого упавший тест нельзя было бы повторить.
//
// Оговорка: сам генератор std::mt19937 стандарт описывает побитово, а вот
// алгоритм std::uniform_real_distribution оставлен на усмотрение реализации.
// Поэтому массив совпадает между запусками на одном компиляторе, но GCC и Clang
// с libc++ могут выдать разные числа при одинаковом seed.
std::vector<double> random_doubles(std::size_t count, double min, double max,
                                   unsigned seed)
{
    std::mt19937 engine(seed);
    std::uniform_real_distribution<double> distribution(min, max);

    std::vector<double> values;
    values.reserve(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        values.push_back(distribution(engine));
    }

    return values;
}

// Записывает значения в CSV с одним признаком x и загружает статистику по нему.
// Значения пишутся через std::format("{}") — это кратчайшая запись, которая при
// чтении даёт ровно то же double, так что тест сравнивает с исходными числами.
NumericFeatureInfo statistics_of(const std::vector<std::string>& cells, const char* test_name)
{
    std::string text = "ID,x\n";
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        text += std::format("{},{}\n", i, cells[i]);
    }

    const auto path = fixtures::write_file(fixtures::temp_dir(test_name) / "data.csv", text);
    const Dataset dataset(path, fixtures::default_config());
    const DatasetInfo info = dataset.analyze();

    const auto* numeric = std::get_if<NumericFeatureInfo>(&info.features.at("x"));
    CHECK(numeric != nullptr);
    return numeric ? *numeric : NumericFeatureInfo{};
}

int histogram_total(const NumericFeatureInfo& stat)
{
    return std::accumulate(std::begin(stat.values), std::end(stat.values), 0);
}

// --- Известный массив ---------------------------------------------------------

// Числа 1..20 вперемешку и три пропуска. Ответ посчитан вручную по определению
// квантиля из реализации: q-квантиль — это k-й по возрастанию элемент, где
// k = floor(n·q), нумерация с единицы. При n = 20:
//
//   q05:     k = 1   →  1
//   q25:     k = 5   →  5
//   медиана: k = 10  → 10   (при чётном n берётся нижний из двух средних)
//   q75:     k = 15  → 15
//   q95:     k = 19  → 19
//
//   среднее   = (1 + … + 20) / 20 = 10.5
//   дисперсия = Σ(x − 10.5)² / (n − 1) = 665 / 19 = 35   (деление на n − 1, как в реализации)
//
// Гистограмма: 10 интервалов шириной (20 − 1) / 10 = 1.9, границы не совпадают
// ни с одним целым, поэтому в каждый интервал попадают ровно два числа.
void test_known_array()
{
    const std::vector<std::string> cells = {
        "7", "19", "NA", "3", "12", "20", "1", "\"NA\"", "15", "8", "11",
        "4", "17", "NA", "2", "10", "14", "6", "18", "9", "16", "13", "5",
    };
    const NumericFeatureInfo stat = statistics_of(cells, "calculate_known");

    CHECK(stat.missing_count == 3);
    CHECK(stat.min == 1.0);
    CHECK(stat.max == 20.0);

    CHECK(stat.q05 == 1.0);
    CHECK(stat.q25 == 5.0);
    CHECK(stat.median == 10.0);
    CHECK(stat.q75 == 15.0);
    CHECK(stat.q95 == 19.0);

    CHECK_NEAR(stat.mean, 10.5, 1e-12);
    CHECK_NEAR(stat.variance, 35.0, 1e-9);

    CHECK_NEAR(stat.ranges[0].first, 1.0, 1e-12);
    CHECK_NEAR(stat.ranges[9].second, 20.0, 1e-9);
    for (int i = 0; i < 10; ++i)
    {
        CHECK(stat.values[i] == 2);
    }
}

// --- Свойства на случайном массиве ---------------------------------------------

// Эти соотношения верны при любых данных и любом разумном определении квантиля.
// Они не проверяют точные числа, зато ловят грубые ошибки: перепутанные индексы,
// потерянный элемент гистограммы, неотсортированный массив.
void test_random_properties()
{
    const std::vector<double> values = random_doubles(100, 0.0, 10.0, 42);

    std::vector<std::string> cells;
    for (const double value : values)
    {
        cells.push_back(std::format("{}", value));
    }
    const NumericFeatureInfo stat = statistics_of(cells, "calculate_random");

    const auto [min, max] = std::ranges::minmax(values);
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();

    CHECK(stat.missing_count == 0);
    CHECK(stat.min == min);
    CHECK(stat.max == max);

    CHECK(stat.min <= stat.q05);
    CHECK(stat.q05 <= stat.q25);
    CHECK(stat.q25 <= stat.median);
    CHECK(stat.median <= stat.q75);
    CHECK(stat.q75 <= stat.q95);
    CHECK(stat.q95 <= stat.max);

    CHECK_NEAR(stat.mean, mean, 1e-12);
    CHECK(stat.variance >= 0.0);

    // Каждое значение попадает ровно в один интервал, включая максимум.
    CHECK(histogram_total(stat) == static_cast<int>(values.size()));
}

} // namespace

int main()
{
    test_known_array();
    test_random_properties();

    return testing::summary();
}
