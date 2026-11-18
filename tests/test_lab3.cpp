#include "mlcore/models/knn.hpp"
#include "mlcore/models/nw.hpp"
#include "testing.hpp"
#include <limits>
#include <utility>

template <class T>
concept TemporarySlice = requires(T &&x) { std::move(x)(0, std::nullopt); };
static_assert(!TemporarySlice<DatasetMatrix>);
static_assert(!TemporarySlice<const DatasetMatrix>);
static_assert(std::is_same_v<decltype(std::declval<const DatasetMatrix &>()(0, std::nullopt)(0)),
                             const double &>);
std::unique_ptr<IDistance> metric() {
    return std::make_unique<EuclideanDistance>();
}
int main() {
    for (auto layout : {Layout::RowMajor, Layout::ColumnMajor}) {
        DatasetMatrix x(2, 3, layout);
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                x(i, j) = 10 * i + j;
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 3; ++j) {
                CHECK(x(i, std::nullopt)(j) == 10 * i + j);
                CHECK(x(std::nullopt, j)(i) == 10 * i + j);
            }
        const auto row = x(0, std::nullopt);
        row(1) = 77;
        CHECK(x(0, 1) == 77);
        CHECK(row.axis() == SliceAxis::Row);
        CHECK(row.as_const()(1) == 77);
        if (layout == Layout::RowMajor) {
            CHECK((*row).size() == 3);
            CHECK_THROWS_AS(*x(std::nullopt, 0), NonContiguousSliceError);
        } else {
            CHECK_THROWS_AS(*row, NonContiguousSliceError);
            CHECK((*x(std::nullopt, 0)).size() == 2);
        }
        CHECK_THROWS_AS(x(2, std::nullopt), std::out_of_range);
        CHECK_THROWS_AS(row.at(3), std::out_of_range);
        DatasetMatrix empty_row(2, 0, layout), empty_col(0, 3, layout), single(1, 1, layout);
        auto er = empty_row(0, std::nullopt);
        auto ec = empty_col(std::nullopt, 0);
        CHECK(er.data() == nullptr);
        CHECK(ec.data() == nullptr);
        CHECK((*er).empty());
        CHECK((*ec).empty());
        CHECK((*single(0, std::nullopt)).size() == 1);
        CHECK((*single(std::nullopt, 0)).size() == 1);
        DatasetMatrix train(2, 1, layout), query(1, 1, layout);
        train(0, 0) = 0;
        train(1, 0) = 2;
        query(0, 0) = 1;
        std::vector<double> y{0, 10};
        NwRegressor nw(1, metric());
        CHECK_THROWS_AS(nw.predict(query), ModelFitError);
        nw.fit(train, y);
        CHECK_NEAR(nw.predict(query)[0], 5, 1e-12);
        query(0, 0) = 0;
        CHECK_NEAR(nw.predict(query)[0], 10 * std::exp(-2) / (1 + std::exp(-2)), 1e-12);
        NwRegressor tiny(1e-6, metric());
        tiny.fit(train, y);
        query(0, 0) = 0.1;
        CHECK_NEAR(tiny.predict(query)[0], 0, 1e-12);
        nw.fit(train, std::vector<double>{7, 7});
        CHECK_NEAR(nw.predict(query)[0], 7, 1e-12);
        train(1, 0) = 0;
        nw.fit(train, y);
        CHECK_NEAR(nw.predict(query)[0], 5, 1e-12);
        NwClassifier cls(1, metric());
        CHECK_THROWS_AS(cls.classes(), ModelFitError);
        cls.fit(train, std::vector<double>{42, 10});
        auto prob = cls.predict_proba(query);
        CHECK(cls.classes()[0] == 10);
        CHECK_NEAR(prob(0, 0), 0.5, 1e-12);
        CHECK(cls.predict(query)[0] == 10);
        cls.fit(train, std::vector<double>{42, 42});
        CHECK(cls.classes().size() == 1);
        auto one_prob = cls.predict_proba(query);
        CHECK(one_prob(0, 0) == 1);
        DatasetMatrix none(0, 1, layout);
        auto ep = cls.predict_proba(none);
        CHECK(ep.get_rows() == 0);
        CHECK(ep.get_cols() == 1);
        KnnRegressor knn(1, metric());
        knn.fit(train, y);
        CHECK(knn.predict(query)[0] == 0);
        KnnClassifier kc(2, metric(), Weighting::Distance);
        kc.fit(train, std::vector<double>{10, 42});
        auto kp = kc.predict_proba(query);
        CHECK_NEAR(kp(0, 0), 0.5, 1e-12);
        CHECK_THROWS_AS(nw.fit(train, std::vector<double>{1}), SizeError);
        CHECK_THROWS_AS(nw.predict(query), ModelFitError);
        CHECK_THROWS_AS(cls.predict(DatasetMatrix(1, 2)), SizeError);
        query(0, 0) = std::numeric_limits<double>::infinity();
        CHECK_THROWS_AS(cls.predict(query), BadArgument);
        CHECK_THROWS_AS(nw.fit(DatasetMatrix(0, 1), {}), BadArgument);
        CHECK_THROWS_AS(nw.fit(DatasetMatrix(1, 0), std::vector<double>{1}), BadArgument);
    }
    CHECK_THROWS_AS(NwRegressor(0, metric()), BadArgument);
    CHECK_THROWS_AS(NwRegressor(std::numeric_limits<double>::infinity(), metric()), BadArgument);
    CHECK_THROWS_AS(NwRegressor(1, nullptr), BadArgument);
    CHECK_THROWS_AS(MatrixSlice<double>(nullptr, 1, 1, SliceAxis::Row), BadArgument);
    DatasetMatrix huge(1, 1), origin(1, 1);
    huge(0, 0) = std::numeric_limits<double>::max();
    NwRegressor extreme(1, metric());
    extreme.fit(origin, std::vector<double>{1});
    CHECK_THROWS_AS(extreme.predict(huge), BadArgument); // переполнение расстояния явно отклоняется
    CHECK_THROWS_AS(
        extreme.fit(origin, std::vector<double>{std::numeric_limits<double>::quiet_NaN()}),
        BadArgument);
    return testing::summary();
}
