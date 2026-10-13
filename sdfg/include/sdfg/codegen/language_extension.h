#pragma once

#include <string>

#include "sdfg/codegen/utils.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/structure.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace structured_control_flow {
class StructuredLoop;
}

namespace codegen {

/// Loop properties a language may be able to express (e.g. clang loop pragmas in C).
struct LoopHints {
    bool unroll = false;
    bool no_vectorize = false;
};

/// An open region of emitted code, closed by LanguageExtension::end.
struct Body {
    /// Block: the body of a control-flow construct. Scope: a lexical scope that only isolates local names.
    enum class Kind { Block, Scope };

    Kind kind;
    size_t content_mark;
};

class LanguageExtension {
protected:
    sdfg::Function& function_;
    std::string external_prefix_;

public:
    LanguageExtension(sdfg::Function& function, const std::string& external_prefix)
        : function_(function), external_prefix_(external_prefix) {
    }

    virtual ~LanguageExtension() = default;

    const std::string& external_prefix() const {
        return this->external_prefix_;
    }

    virtual const std::string language() const = 0;

    virtual std::string primitive_type(const types::PrimitiveType prim_type) = 0;

    virtual std::string declaration(
        const std::string& name, const types::IType& type, bool use_initializer = false, bool use_alignment = false
    ) = 0;

    virtual std::string type_cast(const std::string& name, const types::IType& type) = 0;

    virtual std::string subset(const types::IType& type, const data_flow::Subset& subset) = 0;

    virtual std::string expression(const symbolic::Expression expr) = 0;

    virtual std::string access_node(const data_flow::AccessNode& node) = 0;

    virtual std::string tasklet(const data_flow::Tasklet& tasklet) = 0;

    virtual std::string zero(const types::PrimitiveType prim_type) = 0;

    /* Statements; each returns one complete statement without a trailing newline. */

    virtual std::string comment(const std::string& text) = 0;

    virtual std::string assignment(const std::string& lhs, const std::string& rhs) = 0;

    /// Declares a local, initialized if @p init is non-empty. Empty if the language needs no declaration.
    virtual std::string
    local_declaration(const std::string& name, const types::IType& type, const std::string& init = "") = 0;

    virtual std::string copy_bytes(const std::string& dst, const std::string& src, const std::string& size) = 0;

    virtual std::string break_statement() = 0;

    virtual std::string continue_statement() = 0;

    virtual std::string return_statement(const std::string& value) = 0;

    /// Emits the release of a container the function owns, if its storage requires one.
    virtual void deallocation(PrettyPrinter& stream, const std::string& container, const types::IType& type) = 0;

    /* Expressions */

    virtual std::string null_literal() = 0;

    virtual std::string address_of(const std::string& expr) = 0;

    virtual std::string dereference(const std::string& expr) = 0;

    virtual std::string size_of(const std::string& expr) = 0;

    virtual std::string reinterpret_cast_expr(const std::string& expr, const std::string& type) = 0;

    /* Structured control flow: every begin_* opens a Body that must be closed by end(). */

    virtual Body
    begin_for(PrettyPrinter& stream, const structured_control_flow::StructuredLoop& loop, const LoopHints& hints) = 0;

    /// An unconditional loop, left only through break or return.
    virtual Body begin_while(PrettyPrinter& stream) = 0;

    /// Branch @p index of @p count branches of a conditional, guarded by @p condition.
    virtual Body begin_branch(PrettyPrinter& stream, size_t index, size_t count, const symbolic::Condition& condition) = 0;

    virtual Body begin_scope(PrettyPrinter& stream) = 0;

    virtual void end(PrettyPrinter& stream, const Body& body) = 0;
};

} // namespace codegen
} // namespace sdfg
