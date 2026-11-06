#pragma once

#include "mlcore/dataset.hpp"

int show_stat(const DatasetInfo& info, std::size_t n);
void show_info(const DatasetInfo& info);
void show_matrix_pair(std::span<const double> one, std::span<const double> two, TargetType type, std::optional<std::size_t> length = std::nullopt);