#pragma once

#include <exception>
#include <string>

#include "sdfg/element.h"

namespace docc {
namespace lifting {

class NotImplementedException : public std::exception {
private:
    std::string message_;
    sdfg::DebugInfo dbg_info_;
    std::string element_;

public:
    NotImplementedException(const std::string& message, sdfg::DebugInfo dbg_info, const std::string& element)
        : message_(message), dbg_info_(dbg_info), element_(element) {}

    const char* what() const noexcept override { return message_.c_str(); }

    const sdfg::DebugInfo& debug_info() const { return dbg_info_; }

    const std::string& element() const { return element_; }
};

} // namespace lifting
} // namespace docc
