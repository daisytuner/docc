#pragma once

#include <sdfg/structured_sdfg.h>

void dump_sdfg(const sdfg::StructuredSDFG& sdfg, const std::string& step);


std::optional<std::filesystem::path> get_test_output_dir();
