#pragma once

#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

/// Statement and control-flow syntax shared by the C family (C, C++, CUDA, HIP).
class CStyleLanguageExtension : public LanguageExtension {
public:
    using LanguageExtension::LanguageExtension;

    std::string comment(const std::string& text) override;

    std::string assignment(const std::string& lhs, const std::string& rhs) override;

    std::string local_declaration(const std::string& name, const types::IType& type, const std::string& init = "") override;

    std::string copy_bytes(const std::string& dst, const std::string& src, const std::string& size) override;

    std::string break_statement() override;

    std::string continue_statement() override;

    std::string return_statement(const std::string& value) override;

    void deallocation(PrettyPrinter& stream, const std::string& container, const types::IType& type) override;

    std::string null_literal() override;

    std::string address_of(const std::string& expr) override;

    std::string dereference(const std::string& expr) override;

    std::string size_of(const std::string& expr) override;

    std::string reinterpret_cast_expr(const std::string& expr, const std::string& type) override;

    Body begin_for(
        PrettyPrinter& stream, const structured_control_flow::StructuredLoop& loop, const LoopHints& hints
    ) override;

    Body begin_while(PrettyPrinter& stream) override;

    Body begin_branch(PrettyPrinter& stream, size_t index, size_t count, const symbolic::Condition& condition) override;

    Body begin_scope(PrettyPrinter& stream) override;

    void end(PrettyPrinter& stream, const Body& body) override;

protected:
    Body open(PrettyPrinter& stream, Body::Kind kind);
};

} // namespace codegen
} // namespace sdfg
