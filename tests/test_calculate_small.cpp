// Статистика на короткой выборке.
//
// Числовым признак становится при 11 и более различных значениях, а квантиль
// берётся как элемент с номером floor(n·q). Для q05 при n < 20 это floor(<1) = 0,
// и индекс k − 1 становится −1, то есть обращение за пределы массива.
//
// Тест вынесен в отдельную программу: если ошибка есть, программа аварийно
// останавливается, и это не должно мешать остальным проверкам.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <format>
#include <string>
#include <variant>

int main()
{
    // 12 различных чисел — минимум, при котором признак уже числовой, а q05
    // по текущему определению указывает на элемент с номером 0.
    std::string text = "ID,x\n";
    for (int i = 1; i <= 12; ++i)
    {
        text += std::format("{},{}\n", i, i * 10);
    }
    const auto path = fixtures::write_file(fixtures::temp_dir("calculate_small") / "data.csv", text);

    const Dataset dataset(path, fixtures::default_config());
    const DatasetInfo info = dataset.analyze();

    const auto* stat = std::get_if<NumericFeatureInfo>(&info.features.at("x"));
    CHECK(stat != nullptr);
    if (stat)
    {
        // Какое бы определение ни было выбрано, квантиль — одно из значений
        // выборки или точка между ними, то есть лежит в [min, max].
        CHECK(stat->min <= stat->q05 && stat->q05 <= stat->max);
        CHECK(stat->min <= stat->q95 && stat->q95 <= stat->max);
    }

    return testing::summary();
}
