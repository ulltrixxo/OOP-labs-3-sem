#include "mlcore/config.hpp"
#include <string>

bool Config::in_missing_values(const std::string& value) const {
    for (const std::string& missing_value : missing_values) {
        if (value == missing_value) return true;
    }
    return false;
}

bool Config::unnecessary_column(const std::string& value) const {
    for (const std::string& missing_value : unnecessary_columns) {
        if (value == missing_value) return true;
    }
    return false;
}

bool Config::in_targets(const std::string& value) const {
    for (const std::string& target : targets) {
        if (value == target) return true;
    }
    return false;
}
