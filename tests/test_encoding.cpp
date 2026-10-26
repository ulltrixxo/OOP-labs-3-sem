// Предобработка целиком: импутация, стандартизация, one-hot, сборка матрицы.
//
// Обязательная проверка из условия работы 1: корректность кодирования категорий.
// Проверяется не только форма матрицы, но и смысл: в столбце color=A единица
// стоит именно у тех объектов, у которых в CSV было A.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <string>
#include <variant>
#include <vector>

namespace
{

constexpr int kRows = 21;

// Значения признаков для строки r. Числовой признак num: 10, 20, …, 200 и
// пропуск в последней строке. Категориальный color: B, C, A, пропуск по кругу,
// один из пропусков записан в кавычках. Бинарный flag без пропусков.
std::string num_cell(int r)   { return r == kRows - 1 ? "NA" : std::to_string(10 * (r + 1)); }
std::string flag_cell(int r)  { return r % 2 == 0 ? "no" : "yes"; }
std::string color_cell(int r)
{
    switch (r % 4)
    {
        case 0:  return "B";
        case 1:  return "C";
        case 2:  return "A";
        default: return r == 7 ? "\"NA\"" : "NA";
    }
}

// Строки CSV берутся в порядке, заданном order, — чтобы проверить, что
// результат кодирования не зависит от порядка строк в файле.
std::string make_csv(const std::vector<int>& order)
{
    std::string text = "ID,num,color,flag\n";
    for (const int r : order)
    {
        text += std::format("{},{},{},{}\n", r, num_cell(r), color_cell(r), flag_cell(r));
    }
    return text;
}

std::vector<int> rows_in_order(int first)
{
    std::vector<int> order;
    for (int i = 0; i < kRows; ++i)
    {
        order.push_back((first + i) % kRows);
    }
    return order;
}

// Весь конвейер предобработки: от CSV до матрицы и откликов.
ModelMatrix preprocess_with(const std::string& text, const char* test_name, const Config& config)
{
    const auto path = fixtures::write_file(fixtures::temp_dir(test_name) / "data.csv", text);
    Dataset dataset(path, config);
    DatasetInfo info = dataset.analyze();
    dataset.impute(info);
    dataset.scale_numeric_features(info);
    dataset.OneHot(info);
    return dataset.encode();
}

// То же без откликов: все столбцы — признаки, нужна только матрица.
DatasetMatrix preprocess(const std::string& text, const char* test_name)
{
    return preprocess_with(text, test_name, fixtures::default_config()).get_data();
}

std::vector<std::string> names_of(const DatasetMatrix& matrix)
{
    return matrix.get_feature_names();
}

// --- Имена и порядок столбцов ---------------------------------------------------

// Исходный порядок признаков сохраняется, категории внутри признака
// отсортированы, столбец пропуска идёт последним и есть только у признака,
// где пропуски были.
const std::vector<std::string> kExpectedNames = {
    "num",
    "color=A", "color=B", "color=C", "color=__MISSING__",
    "flag=no", "flag=yes",
};

void test_column_names()
{
    DatasetMatrix matrix = preprocess(make_csv(rows_in_order(0)), "encoding_names");

    CHECK(matrix.get_rows() == static_cast<std::size_t>(kRows));
    CHECK(matrix.get_cols() == kExpectedNames.size());
    CHECK(names_of(matrix) == kExpectedNames);
}

// --- Значения индикаторов ---------------------------------------------------------

void test_indicators()
{
    DatasetMatrix matrix = preprocess(make_csv(rows_in_order(0)), "encoding_indicators");
    if (names_of(matrix) != kExpectedNames)
    {
        CHECK(names_of(matrix) == kExpectedNames);   // остальное без этого бессмысленно
        return;
    }

    for (int r = 0; r < kRows; ++r)
    {
        const std::size_t row = static_cast<std::size_t>(r);

        // Какой столбец color должен быть единичным для этой строки.
        std::size_t expected_color = 4;   // color=__MISSING__
        if (color_cell(r) == "A") expected_color = 1;
        if (color_cell(r) == "B") expected_color = 2;
        if (color_cell(r) == "C") expected_color = 3;

        for (std::size_t c = 1; c <= 4; ++c)
        {
            CHECK(matrix(row, c) == (c == expected_color ? 1.0 : 0.0));
        }

        const bool yes = flag_cell(r) == "yes";
        CHECK(matrix(row, 5) == (yes ? 0.0 : 1.0));   // flag=no
        CHECK(matrix(row, 6) == (yes ? 1.0 : 0.0));   // flag=yes
    }
}

// --- Импутация и стандартизация ---------------------------------------------------

// Статистика num по 20 известным значениям 10, 20, …, 200, до заполнения пропуска:
//   среднее = 105;
//   дисперсия = Σ(x − 105)² / 19 = 66500 / 19 = 3500, σ = √3500;
//   медиана — 10-й по возрастанию элемент (k = floor(20·0.5)) = 100.
//
// Пропуск в последней строке заполняется медианой 100, после чего все значения
// стандартизуются по статистике, посчитанной ДО заполнения, как требует условие.
void test_numeric_column()
{
    DatasetMatrix matrix = preprocess(make_csv(rows_in_order(0)), "encoding_numeric");
    if (names_of(matrix) != kExpectedNames)
    {
        CHECK(names_of(matrix) == kExpectedNames);
        return;
    }

    const double mean = 105.0;
    const double sigma = std::sqrt(3500.0);

    for (int r = 0; r < kRows - 1; ++r)
    {
        CHECK_NEAR(matrix(static_cast<std::size_t>(r), 0), (10.0 * (r + 1) - mean) / sigma, 1e-12);
    }
    CHECK_NEAR(matrix(kRows - 1, 0), (100.0 - mean) / sigma, 1e-12);
}

// --- Детерминированность ------------------------------------------------------------

// Тот же файл, обработанный дважды, даёт одинаковую матрицу. Файл с теми же
// строками в другом порядке даёт те же столбцы в том же порядке: номер столбца
// зависит от набора категорий, а не от того, какая встретилась первой.
void test_determinism()
{
    const std::string text = make_csv(rows_in_order(0));
    DatasetMatrix first  = preprocess(text, "encoding_det_1");
    DatasetMatrix second = preprocess(text, "encoding_det_2");

    CHECK(names_of(first) == names_of(second));
    CHECK(first.get_rows() == second.get_rows() && first.get_cols() == second.get_cols());
    if (first.get_rows() == second.get_rows() && first.get_cols() == second.get_cols())
    {
        for (std::size_t r = 0; r < first.get_rows(); ++r)
        {
            for (std::size_t c = 0; c < first.get_cols(); ++c)
            {
                CHECK(first(r, c) == second(r, c));
            }
        }
    }

    // Первая строка теперь с цветом C, а не B: порядок первого появления другой.
    DatasetMatrix rotated = preprocess(make_csv(rows_in_order(1)), "encoding_det_rotated");
    CHECK(names_of(rotated) == kExpectedNames);
}

// --- Отклики ---------------------------------------------------------------------

// Два отклика: категориальный label и числовой score. У трёх строк нет хотя бы
// одного из них: r = 3 (нет label), r = 10 (нет score), r = 15 (нет обоих).
// Такие строки исключаются целиком — и из матрицы признаков, и из всех откликов.
std::string label_cell(int r)
{
    if (r == 3 || r == 15) return r == 15 ? "\"NA\"" : "NA";
    const char* labels[] = {"yes", "no", "maybe"};
    return labels[r % 3];
}
std::string score_cell(int r) { return r == 10 || r == 15 ? "NA" : std::to_string(r * 0.5); }
bool has_targets(int r)       { return r != 3 && r != 10 && r != 15; }

void test_targets()
{
    std::string text = "ID,num,color,label,score\n";
    for (int r = 0; r < kRows; ++r)
    {
        text += std::format("{},{},{},{},{}\n", r, 10 * (r + 1), color_cell(r), label_cell(r), score_cell(r));
    }
    Config config = fixtures::default_config();
    config.targets = {"label", "score"};
    const ModelMatrix result = preprocess_with(text, "encoding_targets", config);
    const DatasetMatrix& data = result.get_data();
    const Targets& targets = result.get_targets();

    std::vector<int> kept;
    for (int r = 0; r < kRows; ++r) if (has_targets(r)) kept.push_back(r);

    // Отклики не попадают в признаки ни в каком виде.
    const std::vector<std::string> names = names_of(data);
    for (const std::string& name : names)
    {
        CHECK(!name.starts_with("label") && !name.starts_with("score"));
    }

    // Тип отклика: label категориальный, score числовой.
    CHECK(targets.contains("label") && targets.contains("score"));
    if (!targets.contains("label") || !targets.contains("score")) return;
    const auto* label = std::get_if<CategorialTarget>(&targets.at("label"));
    const auto* score = std::get_if<NumericTarget>(&targets.at("score"));
    CHECK(label != nullptr && score != nullptr);
    if (!label || !score) return;

    // Строки без ответа исключены; все векторы откликов одной длины с матрицей.
    CHECK(data.get_rows() == kept.size());
    CHECK(label->values.size() == kept.size());
    CHECK(score->values.size() == kept.size());

    // Коды категориального отклика раздаются по алфавиту меток, а пропуск не
    // становится категорией: отклик не импутируется.
    const auto& codes = label->categorical_values;
    CHECK(codes.size() == 3);
    CHECK(codes.contains("maybe") && codes.at("maybe") == 0);
    CHECK(codes.contains("no")    && codes.at("no") == 1);
    CHECK(codes.contains("yes")   && codes.at("yes") == 2);
    CHECK(!codes.contains("__MISSING__"));

    // Каждая строка матрицы и её ответы относятся к одному объекту. Признак
    // проверяется по столбцу индикатора цвета: он однозначно восстанавливает
    // исходную строку CSV, так что сдвиг строк сразу будет виден.
    const auto column = [&](const std::string& name) {
        return static_cast<std::size_t>(std::find(names.begin(), names.end(), name) - names.begin());
    };
    if (data.get_rows() != kept.size() || label->values.size() != kept.size()
        || score->values.size() != kept.size()) return;
    for (std::size_t i = 0; i < kept.size(); ++i)
    {
        const int r = kept[i];
        CHECK(label->values[i] == codes.at(label_cell(r)));
        CHECK(score->values[i] == r * 0.5);   // числовой отклик не масштабирован

        const std::string color = color_cell(r);
        const std::string expected = (color == "NA" || color == "\"NA\"") ? "color=__MISSING__" : "color=" + color;
        CHECK(column(expected) < names.size() && data(i, column(expected)) == 1.0);
    }
}

} // namespace

int main()
{
    test_column_names();
    test_indicators();
    test_numeric_column();
    test_determinism();
    test_targets();

    return testing::summary();
}
