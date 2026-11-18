#pragma once
#include "mlcore/config.hpp"

// Общие настройки исходного датасета: lab1 использует их для подготовки,
// lab3 — для выбора откликов из сохранённого файла.
namespace app_config {
inline const Config dataset{
    .delimiter = ',',
    .missing_values = {"NA", "\"NA\""},
    .unnecessary_columns = {"ID"},
    .impute_missing_value = "__MISSING__",
    .targets = {"Status", "N_Days"}
};
}
