#include "testing.hpp"
#include "oop/dataset.hpp"
#include <cstddef>
#include <random>
#include <vector>
#include <algorithm>
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

} // namespace

int main()
{
    std::vector<double> values = random_doubles(100, 0.0, 10.0, 42);

    NumericFeatureInfo result = calculate_num_arr(values);
    auto [min1, max1] = std::ranges::minmax(values);
    std::size_t sum_gist = 0;
    for (auto x : result.values) {
        sum_gist += x;
    }
    CHECK(min1 == result.min);
    CHECK(max1 == result.max);
    CHECK(result.min <= result.q05);
    CHECK(result.q05 <= result.q25);
    CHECK(result.q25 <= result.median);
    CHECK(result.median <= result.q75);
    CHECK(result.q75 <= result.q95);
    CHECK(result.q95 <= result.max);
    CHECK(result.mean >= result.min);
    CHECK(result.mean <= result.max);
    CHECK(result.variance >= 0);
    CHECK(sum_gist == values.size());
    return testing::summary();
}
