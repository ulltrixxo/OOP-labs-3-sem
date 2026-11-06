#include "testing.hpp"
#include "fixtures.hpp"
#include "mlcore/model_matrix.hpp"
#include "mlcore/exceptions.hpp"
#include <fstream>
int main() {
    DatasetMatrix x(3,1,Layout::RowMajor,{"x"});
    for (std::size_t i=0;i<3;++i) x(i,0)=i+0.5;
    Targets targets{{"N_Days",{TargetType::Numeric,{400,4500,1012},{}}},
                    {"Status",{TargetType::Categorical,{2,0,1},{{"C",0},{"CL",1},{"D",2}}}}};
    auto file=fixtures::temp_dir("targets_binary")/"roundtrip.bin";
    save(ModelMatrix(x,targets),file);
    auto loaded=ModelMatrix::load(file);
    CHECK(loaded.get_targets().size()==2);
    for (const auto& [name,target] : targets) {
        const auto& actual=loaded.get_targets().at(name);
        CHECK(actual.type==target.type);
        CHECK(actual.values==target.values);
        CHECK(actual.categorical_values==target.categorical_values);
    }
    for (std::size_t i=0;i<3;++i) CHECK(loaded.get_data()(i,0)==x(i,0));
    // Truncate a valid file at every byte boundary, covering all payload fields.
    std::ifstream in(file,std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(in)),{});
    for (std::size_t n=0;n<bytes.size();++n) {
        auto broken=fixtures::write_file(file.parent_path()/"short.bin",bytes.substr(0,n));
        CHECK_THROWS_AS(ModelMatrix::load(broken),FileReadError);
    }
    return testing::summary();
}
