#pragma once

#include <symengine/printers/codegen.h>

#include <string>
#include <vector>

#include "sdfg/codegen/language_extension.h"

namespace sdfg {
namespace codegen {

/**
 * Emits Python operating on NumPy arrays and scalars with C semantics: values are cast to their
 * NumPy dtype when read, integer division truncates, and pointers are numpy arrays that can be
 * indexed but not taken the address of or dereferenced.
 */
class PythonLanguageExtension : public LanguageExtension {
private:
    struct OpenBody {
        Body::Kind kind;
        bool is_loop;
        std::string loop_update;
    };

    std::vector<OpenBody> open_bodies_;

    std::string literal(const std::string& data, const types::IType& type);

    Body open(PrettyPrinter& stream, Body::Kind kind, bool is_loop, const std::string& loop_update = "");

public:
    PythonLanguageExtension(sdfg::Function& function, const std::string& external_prefix = "")
        : LanguageExtension(function, external_prefix) {
    }

    /// Imports and helper functions that generated code relies on.
    static std::string runtime_prelude();

    const std::string language() const override {
        return "Python";
    }

    std::string primitive_type(const types::PrimitiveType prim_type) override;

    std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) override;

    std::string type_cast(const std::string& name, const types::IType& type) override;

    std::string subset(const types::IType& type, const data_flow::Subset& subset) override;

    std::string expression(const symbolic::Expression expr) override;

    std::string access_node(const data_flow::AccessNode& node) override;

    std::string tasklet(const data_flow::Tasklet& tasklet) override;

    std::string zero(const types::PrimitiveType prim_type) override;

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
};

class PythonSymbolicPrinter : public SymEngine::BaseVisitor<PythonSymbolicPrinter, SymEngine::CodePrinter> {
private:
    sdfg::Function& function_;

public:
    using SymEngine::CodePrinter::apply;
    using SymEngine::CodePrinter::bvisit;
    using SymEngine::CodePrinter::str_;

    void bvisit(const SymEngine::Infty& x);
    void bvisit(const SymEngine::BooleanAtom& x);
    void bvisit(const SymEngine::Symbol& x);

    void bvisit(const SymEngine::And& x);
    void bvisit(const SymEngine::Or& x);
    void bvisit(const SymEngine::Not& x);
    void bvisit(const SymEngine::Equality& x);
    void bvisit(const SymEngine::Unequality& x);

    void bvisit(const SymEngine::Min& x);
    void bvisit(const SymEngine::Max& x);
    void bvisit(const SymEngine::FunctionSymbol& x);

    void _print_pow(
        std::ostringstream& o,
        const SymEngine::RCP<const SymEngine::Basic>& a,
        const SymEngine::RCP<const SymEngine::Basic>& b
    ) override;

    explicit PythonSymbolicPrinter(sdfg::Function& function) : function_(function) {
    }
};

} // namespace codegen
} // namespace sdfg
