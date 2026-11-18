#include "../common/dataset_config.hpp"
#include "mlcore/model_matrix.hpp"
#include "mlcore/models/knn.hpp"
#include "mlcore/models/nw.hpp"
#include "../common/visualize.hpp"
#include <algorithm>
#include <format>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        const bool use_nw = true; // выбор модели
        const double bandwidth = 2; // h
        const std::size_t k = 5;
        const std::size_t preview_rows = 10; // количество строк таблицы в консоли
        const Config& config = app_config::dataset;
        ModelMatrix data = ModelMatrix::load("data/cirrhosis.bin");
        Split split = data.split(0.7, 1);

        // Конфиг и сохранённый файл должны относиться к одному датасету.
        if (config.targets.empty()) throw BadArgument("No targets configured");

        for (const std::string& name : config.targets)
            if (!data.get_targets().contains(name))
                throw BadArgument(std::format(
                    "Configured target '{}' is missing from the binary dataset; regenerate it with lab1", name));

        // начало программы
        std::unique_ptr<IRegressor> reg;
        std::unique_ptr<IClassifier> cls;
        if (use_nw) {
            reg = std::make_unique<NwRegressor>(bandwidth, std::make_unique<EuclideanDistance>());
            cls = std::make_unique<NwClassifier>(bandwidth, std::make_unique<EuclideanDistance>());
        } else {
            reg = std::make_unique<KnnRegressor>(k, std::make_unique<EuclideanDistance>());
            cls = std::make_unique<KnnClassifier>(k, std::make_unique<EuclideanDistance>());
        }
        std::cout << (use_nw ? std::format("Model: NW   h = {:.3g}\n", bandwidth)
                            : std::format("Model: kNN  k = {}\n", k));

        for (const std::string& name : config.targets) {

            const Target& target = data.get_targets().at(name);
            const std::span<const double> train_data = split.train.target_values(name);
            const std::span<const double> test_data = split.test.target_values(name);
            const std::size_t count = std::min(preview_rows, test_data.size()); // для вывода
            
            std::vector<std::vector<std::string>> rows;
            switch (target.type) {
                case TargetType::Numeric: {
                    reg->fit(split.train.get_data(), train_data);
                    const std::vector<double> predicted = reg->predict(split.test.get_data());
                    std::cout << std::format("\nRegression: {}\n", name);
                    for (std::size_t i = 0; i < count; ++i)
                        rows.push_back({std::format("{}", i + 1), std::format("{:.2f}", test_data[i]),
                                        std::format("{:.2f}", predicted[i])});
                    print_table({"#", "Actual", "Predicted"}, rows);
                    break;
                }
                case TargetType::Categorical: {
                    cls->fit(split.train.get_data(), train_data);
                    const std::vector<double> predicted = cls->predict(split.test.get_data());
                    const DatasetMatrix probabilities = cls->predict_proba(split.test.get_data());
                    const std::span<const double> classes = cls->classes();
                    std::cout << std::format("\nClassification: {}\nClasses:", name);
                    std::vector<std::string> headers{"#", "Actual", "Predicted"};
                    for (double code : classes) {
                        const std::string label = class_name(code, target);
                        std::cout << std::format(" {:g}={}", code, label);
                        headers.push_back(std::format("P({})", label));
                    }
                    std::cout << '\n';
                    for (std::size_t i = 0; i < count; ++i) {
                        std::vector<std::string> row{std::format("{}", i + 1),
                            class_name(test_data[i], target), class_name(predicted[i], target)};
                        for (std::size_t c = 0; c < classes.size(); ++c)
                            row.push_back(std::format("{:.4f}", probabilities(i, c)));
                        rows.push_back(std::move(row));
                    }
                    print_table(headers, rows);
                    break;
                }
                default: {
                    throw BadArgument(std::format("Unknown target type for '{}'", name));
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }
}
