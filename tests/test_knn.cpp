// Independent small examples, a sorting oracle, and public API error contracts.
#include "testing.hpp"
#include "mlcore/models/knn.hpp"
#include "mlcore/exceptions.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

DatasetMatrix matrix(std::initializer_list<double> values) {
    DatasetMatrix m(values.size(), 1, Layout::RowMajor, {"x"});
    std::size_t row = 0;
    for (double x : values) m(row++, 0) = x;
    return m;
}
std::unique_ptr<IDistance> metric() { return std::make_unique<EuclideanDistance>(); }
// Expose the neighbours through the existing aggregation extension point.
class Probe : public kNNBase {
    double consolidate(const std::vector<Neighbour>& neighbours) const override {
        seen.clear();
        for (auto n : neighbours) seen.push_back(n.index);
        std::sort(seen.begin(), seen.end());
        return 0;
    }
public:
    mutable std::vector<std::size_t> seen;
    explicit Probe(std::size_t k) : kNNBase(k, metric()) {}
};
int main() {
    const std::array<double, 2> a{0, 0}, b{3, 4};
    EuclideanDistance eu; ManhattanDistance man; ChebyshevDistance cheb;
    CHECK_NEAR(eu.compute(a,b),5,1e-12);
    CHECK_NEAR(man.compute(a,b),7,1e-12);
    CHECK_NEAR(cheb.compute(a,b),4,1e-12);
    for (const IDistance* d : std::array<const IDistance*,3>{&eu,&man,&cheb}) {
        CHECK_NEAR(d->compute(a,a),0,1e-12);
        CHECK_THROWS_AS(d->compute(a, std::span<const double>(b).first(1)),SizeError);
    }
    const auto train = matrix({0,2,4});
    const std::vector<double> y{0,10,20};
    kNNRegressor reg(3,metric()); reg.fit(train,y);
    CHECK_NEAR(reg.predict(matrix({1}))[0],10,1e-12);
    kNNRegressor weighted(3,metric(),Weighting::Distance); weighted.fit(train,y);
    // Distances 1,1,3: normalized weighted sum = 50/7.
    CHECK_NEAR(weighted.predict(matrix({1}))[0],50.0/7,1e-12);
    CHECK_NEAR(weighted.predict(matrix({2}))[0],10,1e-12);
    for (auto mode : {Weighting::Uniform,Weighting::Distance}) {
        kNNRegressor identity(1,metric(),mode); identity.fit(train,y);
        CHECK(identity.predict(train)==y);
        kNNClassifier classes(1,metric(),mode); classes.fit(train,y);
        CHECK(classes.predict(train)==y);
    }
    // Close minority class wins only with distance weighting.
    const auto cl = matrix({1,3,4}); const std::vector<double> labels{2,1,1};
    kNNClassifier uniform(3,metric()); uniform.fit(cl,labels);
    kNNClassifier distance(3,metric(),Weighting::Distance); distance.fit(cl,labels);
    CHECK(uniform.predict(matrix({0}))[0]==1);
    CHECK(distance.predict(matrix({0}))[0]==2);
    // Duplicate exact matches: average or vote using only zero-distance rows.
    const auto duplicates=matrix({0,0,5}); const std::vector<double> dy{2,4,100};
    weighted.fit(duplicates,dy); distance.fit(duplicates,dy);
    CHECK_NEAR(weighted.predict(matrix({0}))[0],3,1e-12);
    CHECK(distance.predict(matrix({0}))[0]==2); // vote tie -> smaller label
    // Sorting oracle includes distance ties and every legal k.
    const auto ties=matrix({-2,2,-1,1,5}); const std::vector<double> ty(5,0);
    for (std::size_t k=1;k<=5;++k) {
        Probe probe(k); probe.fit(ties,ty);
        for (double q : {-1.0,0.0,2.0}) {
            probe.predict(matrix({q}));
            std::vector<std::pair<double,std::size_t>> sorted;
            for (std::size_t i=0;i<5;++i) sorted.emplace_back(std::abs(ties(i,0)-q),i);
            std::sort(sorted.begin(),sorted.end());
            std::vector<std::size_t> expected;
            for (std::size_t i=0;i<k;++i) expected.push_back(sorted[i].second);
            std::sort(expected.begin(),expected.end());
            CHECK(probe.seen==expected);
        }
    }
    kNNRegressor fresh(1,metric());
    CHECK_THROWS_AS(fresh.predict(train),ModelFitError);
    CHECK_THROWS_AS(kNNRegressor(0,metric()),BadArgument);
    CHECK_THROWS_AS(kNNRegressor(1,nullptr),BadArgument);
    CHECK_THROWS_AS(fresh.fit(matrix({}),{}),BadArgument);
    CHECK_THROWS_AS(reg.fit(matrix({0}),std::vector<double>{0}),BadArgument);
    CHECK_THROWS_AS(fresh.fit(train,std::vector<double>{0}),SizeError);
    CHECK_THROWS_AS(fresh.predict(train),ModelFitError);
    fresh.fit(train,y);
    CHECK_THROWS_AS(fresh.predict(DatasetMatrix(1,2)),SizeError);
    CHECK(fresh.predict(DatasetMatrix(0,1)).empty());
    CHECK_THROWS_AS(fresh.fit(matrix({std::numeric_limits<double>::quiet_NaN()}),std::vector<double>{0}),BadArgument);
    CHECK(fresh.predict(train)==y); // rejected fit preserves the old model
    DatasetMatrix bounds(1,1); const auto& cbounds=bounds;
    CHECK_THROWS_AS(bounds.at(1,0),std::out_of_range);
    CHECK_THROWS_AS(cbounds.at(0,1),std::out_of_range);
    return testing::summary();
}
