#pragma once

namespace docc::compile {
class SrcFileCompilerBuilder;
}

namespace docc::target {

struct DoccTarget {
    std::string short_name;

    bool (*apply_additional_compile_options)(docc::compile::SrcFileCompilerBuilder& src_compile_builder);
};

} // namespace docc::target
