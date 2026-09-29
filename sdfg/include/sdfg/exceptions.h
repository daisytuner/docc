#pragma once

#include <exception>
#include <string>

namespace sdfg {

class InvalidSDFGException : public std::exception {
private:
    std::string message_;

public:
    InvalidSDFGException(const std::string& message) : message_(message) {
    }

    const char* what() const noexcept override {
        return message_.c_str();
    }
};

class UnstructuredControlFlowException : public std::exception {
public:
    const char* what() const noexcept override {
        return "Unstructured control flow detected";
    }
};

class StringEnum {
public:
    StringEnum(const std::string& value) : value_(value) {
    }
    StringEnum(const StringEnum& other) : value_(other.value_) {
    }
    StringEnum(StringEnum&& other) noexcept : value_(std::move(other.value_)) {
    }

    StringEnum& operator=(const StringEnum& other) {
        value_ = other.value_;
        return *this;
    }

    StringEnum& operator=(StringEnum&& other) noexcept {
        value_ = std::move(other.value_);
        return *this;
    }

    std::string value() const {
        return value_;
    }

    bool operator==(const StringEnum& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const StringEnum& other) const {
        return !(*this == other);
    }

private:
    std::string value_;
};

} // namespace sdfg
