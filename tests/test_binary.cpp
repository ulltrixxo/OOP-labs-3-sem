// Бинарный формат: запись и чтение матрицы.
//
// Обязательная проверка из условия работы 1: сохранить матрицу, загрузить её
// обратно и сравнить значения с исходными. Файлы пишутся во временный каталог.

#include "fixtures.hpp"
#include "testing.hpp"

#include "mlcore/dataset.hpp"

#include <cstddef>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

namespace
{

// Путь к файлу задаётся в main, до запуска проверок.
fs::path saved_file;

// Значения подобраны так, чтобы ловить потерю точности при текстовом
// преобразовании: дробные, отрицательные, очень большие и очень малые.
double sample_value(std::size_t r, std::size_t c)
{
    const double special[] = {0.1, -1.0 / 3.0, 1e300, -2.5e-300, 123456.789};
    return (r * 10 + c) % 7 == 0 ? special[(r + c) % 5] : r * 10.0 + c + 0.125;
}

DatasetMatrix make_matrix(std::size_t rows, std::size_t cols)
{
    std::vector<std::string> names;
    for (std::size_t c = 0; c < cols; ++c)
    {
        names.push_back(std::format("feature_{}=значение", c));   // '=' и кириллица
    }

    DatasetMatrix matrix(rows, cols, Layout::RowMajor, names);
    for (std::size_t r = 0; r < rows; ++r)
    {
        for (std::size_t c = 0; c < cols; ++c)
        {
            matrix(r, c) = sample_value(r, c);
        }
    }
    return matrix;
}

// Матрица, прочитанная из файла, совпадает с исходной во всём: размеры, порядок
// хранения, имена столбцов, значения. double сравниваются точно: байты копируются
// без арифметики, поэтому любое отличие — ошибка формата.
void check_round_trip(DatasetMatrix& original)
{
    save(original, saved_file);
    DatasetMatrix loaded = DatasetMatrix::load(saved_file);

    CHECK(loaded.get_rows() == original.get_rows());
    CHECK(loaded.get_cols() == original.get_cols());
    CHECK(loaded.get_layout() == original.get_layout());

    if (loaded.get_rows() != original.get_rows() || loaded.get_cols() != original.get_cols())
    {
        return;
    }

    for (std::size_t c = 0; c < original.get_cols(); ++c)
    {
        CHECK(loaded.get_feature_name(c) == original.get_feature_name(c));
    }
    for (std::size_t r = 0; r < original.get_rows(); ++r)
    {
        for (std::size_t c = 0; c < original.get_cols(); ++c)
        {
            CHECK(loaded(r, c) == original(r, c));
        }
    }
}

// --- Прямоугольные матрицы ------------------------------------------------------

// Две формы, 3×4 и 4×3: на квадратной матрице перепутанные при чтении строки и
// столбцы были бы незаметны.
void test_rectangular()
{
    DatasetMatrix wide = make_matrix(3, 4);
    check_round_trip(wide);

    DatasetMatrix tall = make_matrix(4, 3);
    check_round_trip(tall);
}

// --- Размер файла -----------------------------------------------------------------

// Документирует формат: число столбцов, число строк, порядок хранения, затем для
// каждого имени его длина и символы, затем все значения подряд.
void test_file_size()
{
    DatasetMatrix matrix = make_matrix(3, 4);
    save(matrix, saved_file);

    std::size_t expected = sizeof(std::size_t) * 2 + sizeof(Layout);
    for (std::size_t c = 0; c < matrix.get_cols(); ++c)
    {
        expected += sizeof(std::size_t) + matrix.get_feature_name(c).size();
    }
    expected += matrix.get_rows() * matrix.get_cols() * sizeof(double);

    CHECK(fs::file_size(saved_file) == expected);
}

// --- Матрица из настоящей предобработки -------------------------------------------

void test_pipeline_matrix()
{
    std::string text = "ID,num,color\n";
    for (int i = 0; i < 25; ++i)
    {
        text += std::format("{},{},{}\n", i, i % 6 == 0 ? "NA" : std::to_string(i * 1.5),
                            i % 4 == 0 ? "NA" : (i % 2 == 0 ? "red" : "blue"));
    }
    const auto path = fixtures::write_file(saved_file.parent_path() / "pipeline.csv", text);

    Dataset dataset(path, fixtures::default_config());
    DatasetInfo info = dataset.analyze();
    dataset.impute(info);
    dataset.scale_numeric_features(info);
    dataset.OneHot(info);
    DatasetMatrix matrix = dataset.encode().get_data();

    check_round_trip(matrix);
}

} // namespace

int main()
{
    saved_file = fixtures::temp_dir("binary") / "matrix.bin";

    test_rectangular();
    test_file_size();
    test_pipeline_matrix();

    return testing::summary();
}
