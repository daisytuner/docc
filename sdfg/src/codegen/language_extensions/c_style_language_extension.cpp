#include "sdfg/codegen/language_extensions/c_style_language_extension.h"

#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace codegen {

std::string CStyleLanguageExtension::comment(const std::string& text) {
    return "// " + text;
}

std::string CStyleLanguageExtension::assignment(const std::string& lhs, const std::string& rhs) {
    return lhs + " = " + rhs + ";";
}

std::string CStyleLanguageExtension::
    local_declaration(const std::string& name, const types::IType& type, const std::string& init) {
    std::string decl = this->declaration(name, type);
    if (!init.empty()) {
        decl += " = " + init;
    }
    return decl + ";";
}

std::string CStyleLanguageExtension::copy_bytes(const std::string& dst, const std::string& src, const std::string& size) {
    return "memcpy(" + dst + ", " + src + ", " + size + ");";
}

std::string CStyleLanguageExtension::break_statement() {
    return "break;";
}

std::string CStyleLanguageExtension::continue_statement() {
    return "continue;";
}

std::string CStyleLanguageExtension::return_statement(const std::string& value) {
    return "return " + value + ";";
}

void CStyleLanguageExtension::deallocation(PrettyPrinter& stream, const std::string& container, const types::IType& type) {
    if (type.storage_type().deallocation() != types::StorageType::AllocationType::Managed) {
        return;
    }
    if (type.storage_type().is_cpu_heap()) {
        stream << this->external_prefix() << "free(" << container << ");" << std::endl;
    } else if (type.storage_type().is_nv_generic()) {
        stream << "cudaSetDevice(0);" << std::endl;
        stream << "cudaFree(" << container << ");" << std::endl;
    }
}

std::string CStyleLanguageExtension::null_literal() {
    return "nullptr";
}

std::string CStyleLanguageExtension::address_of(const std::string& expr) {
    return "&" + expr;
}

std::string CStyleLanguageExtension::dereference(const std::string& expr) {
    return "*" + expr;
}

std::string CStyleLanguageExtension::size_of(const std::string& expr) {
    return "sizeof " + expr;
}

std::string CStyleLanguageExtension::reinterpret_cast_expr(const std::string& expr, const std::string& type) {
    return "reinterpret_cast<" + type + ">(" + expr + ")";
}

Body CStyleLanguageExtension::
    begin_for(PrettyPrinter& stream, const structured_control_flow::StructuredLoop& loop, const LoopHints& hints) {
    if (hints.unroll || hints.no_vectorize) {
        stream << "#pragma clang loop";
        if (hints.no_vectorize) {
            stream << " vectorize(disable) interleave(disable)";
        }
        if (hints.unroll) {
            stream << " unroll(full)";
        }
        stream << std::endl;
    }
    auto indvar = loop.indvar()->get_name();
    stream << "for(" << indvar << " = " << this->expression(loop.init()) << ";" << this->expression(loop.condition())
           << ";" << indvar << " = " << this->expression(loop.update()) << ")" << std::endl;
    return open(stream, Body::Kind::Block);
}

Body CStyleLanguageExtension::begin_while(PrettyPrinter& stream) {
    stream << "while (1)" << std::endl;
    return open(stream, Body::Kind::Block);
}

Body CStyleLanguageExtension::
    begin_branch(PrettyPrinter& stream, size_t index, size_t count, const symbolic::Condition& condition) {
    stream << (index == 0 ? "if" : "else if") << "(" << this->expression(condition) << ")" << std::endl;
    return open(stream, Body::Kind::Block);
}

Body CStyleLanguageExtension::begin_scope(PrettyPrinter& stream) {
    return open(stream, Body::Kind::Scope);
}

Body CStyleLanguageExtension::open(PrettyPrinter& stream, Body::Kind kind) {
    stream << "{" << std::endl;
    stream.changeIndent(+4);
    return Body{kind, stream.content()};
}

void CStyleLanguageExtension::end(PrettyPrinter& stream, const Body& body) {
    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

} // namespace codegen
} // namespace sdfg
