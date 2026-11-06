// Лабораторная работа 2. Метод k ближайших соседей.
#include "mlcore/dataset.hpp"
#include "mlcore/models/knn.hpp"
#include "mlcore/matrix.hpp"
#include "mlcore/exceptions.hpp"
#include "visualize.hpp"

#include <exception>
#include <iostream>
#include <memory>
#include <cstddef>
#include <filesystem>

int main() {
    try {
        const std::size_t k = 5; // k ближайщих
        const Weighting weighting = Weighting::Uniform; // вес
        const std::filesystem::path data_path = "data/cirrhosis.bin"; // файл

        const double train_proportion = 0.7;
        const unsigned random_state = 1;
        const std::size_t preview_rows = 10;

        ModelMatrix matrix = ModelMatrix::load(data_path);

        Split split_matrix = matrix.split(train_proportion, random_state);
        std::unique_ptr<IModel> num_model = std::make_unique<kNNRegressor>(k, std::make_unique<EuclideanDistance>(), weighting);
        num_model->fit(split_matrix.train.get_data(), split_matrix.train.target_values("N_Days"));
        std::vector<double> ans1 = num_model->predict(split_matrix.test.get_data());
        show_matrix_pair(split_matrix.test.target_values("N_Days"), ans1, TargetType::Numeric, preview_rows);


        std::unique_ptr<IModel> cat_model = std::make_unique<kNNClassifier>(k, std::make_unique<EuclideanDistance>(), weighting);
        cat_model->fit(split_matrix.train.get_data(), split_matrix.train.target_values("Status"));
        std::vector<double> ans2 = cat_model->predict(split_matrix.test.get_data());
        show_matrix_pair(split_matrix.test.target_values("Status"), ans2, TargetType::Categorical, preview_rows);
    }
    catch (const SizeError& e) {
        std::cerr << "Dimension error: " << e.what() << '\n';
        return 1;
    }
    catch (const BadArgument& e) {
        std::cerr << "Invalid argument: " << e.what() << '\n';
        return 1;
    }
    catch (const ArgumentError& e) {
        std::cerr << "Argument error: " << e.what() << '\n';
        return 1;
    }
    catch (const ModelFitError& e) {
        std::cerr << "Model state error: " << e.what() << '\n';
        return 1;
    }
    catch (const MatrixTypeError& e) {
        std::cerr << "Unsupported matrix layout: " << e.what() << '\n';
        return 1;
    }
    catch (const FileOpenError& e) {
        std::cerr << "File open error: " << e.what() << '\n';
        return 1;
    }
    catch (const FileReadError& e) {
        std::cerr << "File read error: " << e.what() << '\n';
        return 1;
    }
    catch (const FileFormatError& e) {
        std::cerr << "File format error: " << e.what() << '\n';
        return 1;
    }
    catch (const FileError& e) {
        std::cerr << "File error: " << e.what() << '\n';
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
