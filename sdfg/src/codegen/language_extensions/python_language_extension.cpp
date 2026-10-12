#include "sdfg/codegen/language_extensions/python_language_extension.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/structured_loop.h"

namespace sdfg {
namespace codegen {

namespace {

[[noreturn]] void unsupported(const std::string& what) {
    throw std::runtime_error("Python codegen: " + what + " is not supported");
}

std::string join(const std::vector<std::string>& parts, const std::string& delimiter) {
    std::string result;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            result += delimiter;
        }
        result += parts[i];
    }
    return result;
}

bool is_integral(types::PrimitiveType prim) {
    return types::is_integer(prim) || prim == types::PrimitiveType::Bool;
}

/// sizeof in C, where bool occupies a whole byte.
size_t byte_width(types::PrimitiveType prim) {
    return (types::bit_width(prim) + 7) / 8;
}

/// Two's-complement bit pattern of a C integer literal; C suffixes and bases are accepted.
uint64_t integer_bits(const std::string& data) {
    static const std::unordered_map<std::string, uint64_t> named = {
        {"true", 1},
        {"false", 0},
        {"INT8_MIN", static_cast<uint64_t>(INT8_MIN)},
        {"INT8_MAX", INT8_MAX},
        {"UINT8_MAX", UINT8_MAX},
        {"INT16_MIN", static_cast<uint64_t>(INT16_MIN)},
        {"INT16_MAX", INT16_MAX},
        {"UINT16_MAX", UINT16_MAX},
        {"INT32_MIN", static_cast<uint64_t>(INT32_MIN)},
        {"INT32_MAX", INT32_MAX},
        {"UINT32_MAX", UINT32_MAX},
        {"INT64_MIN", static_cast<uint64_t>(INT64_MIN)},
        {"INT64_MAX", INT64_MAX},
        {"UINT64_MAX", UINT64_MAX},
    };
    if (auto it = named.find(data); it != named.end()) {
        return it->second;
    }
    std::string digits = data;
    while (!digits.empty() && std::strchr("uUlL", digits.back())) {
        digits.pop_back();
    }
    if (!digits.empty() && digits[0] == '-') {
        return static_cast<uint64_t>(std::stoll(digits, nullptr, 0));
    }
    return std::stoull(digits, nullptr, 0);
}

std::string integer_value(uint64_t bits, types::PrimitiveType prim) {
    size_t width = types::bit_width(prim);
    if (width < 64) {
        bits &= (uint64_t(1) << width) - 1;
    }
    if (types::is_unsigned(prim)) {
        return std::to_string(bits);
    }
    int64_t value = static_cast<int64_t>(bits);
    if (width < 64 && ((bits >> (width - 1)) & 1)) {
        value -= int64_t(1) << width;
    }
    return std::to_string(value);
}

std::string floating_value(const std::string& data) {
    std::string text = data;
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (lower == "infinity" || lower == "+infinity" || lower == "inf") {
        return "np.inf";
    }
    if (lower == "-infinity" || lower == "-inf") {
        return "-np.inf";
    }
    if (lower == "nan" || lower == "-nan") {
        return "np.nan";
    }
    static const std::unordered_map<std::string, std::string> math_constants = {
        {"M_PI", "math.pi"},
        {"M_PI_2", "(math.pi / 2)"},
        {"M_PI_4", "(math.pi / 4)"},
        {"M_1_PI", "(1 / math.pi)"},
        {"M_2_PI", "(2 / math.pi)"},
        {"M_2_SQRTPI", "(2 / math.sqrt(math.pi))"},
        {"M_E", "math.e"},
        {"M_LOG2E", "math.log2(math.e)"},
        {"M_LOG10E", "math.log10(math.e)"},
        {"M_LN2", "math.log(2)"},
        {"M_LN10", "math.log(10)"},
        {"M_SQRT2", "math.sqrt(2)"},
        {"M_SQRT1_2", "math.sqrt(0.5)"},
    };
    if (auto it = math_constants.find(text); it != math_constants.end()) {
        return it->second;
    }
    while (!text.empty() && std::strchr("fFlL", text.back())) {
        text.pop_back();
    }
    char* end = nullptr;
    double value = std::strtod(text.c_str(), &end);
    if (text.empty() || *end != '\0') {
        throw std::runtime_error("Python codegen: invalid floating-point literal " + data);
    }
    if (std::isnan(value)) {
        return "np.nan";
    }
    if (std::isinf(value)) {
        return value > 0 ? "np.inf" : "-np.inf";
    }
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

} // namespace

std::string PythonLanguageExtension::runtime_prelude() {
    return R"(import math

import ml_dtypes
import numpy as np


def _docc_idiv(a, b):
    a, b = int(a), int(b)
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _docc_imod(a, b):
    a, b = int(a), int(b)
    return a - b * _docc_idiv(a, b)


def _docc_sdiv(a, b):
    return type(a)(_docc_idiv(a, b))


def _docc_srem(a, b):
    return type(a)(_docc_imod(a, b))


def _docc_zext_i64(x):
    return int(np.int64(np.uint64(np.int64(int(x)))))


def _docc_trunc_i32(x):
    return int(np.int32(np.uint32(np.uint64(np.int64(int(x))))))


def _docc_cast(dtype, value):
    # NumPy rejects out-of-range Python ints; NumPy scalars convert with C wraparound.
    if type(value) is int and np.issubdtype(dtype, np.integer):
        value = np.uint64(value & 0xFFFFFFFFFFFFFFFF)
    return dtype(value)


def _docc_bytes(buffer):
    return buffer.reshape(-1).view(np.uint8)


def _docc_memcpy(dst, src, count):
    count = int(count)
    _docc_bytes(dst)[:count] = _docc_bytes(src)[:count]


def _docc_memset(dst, value, count):
    _docc_bytes(dst)[: int(count)] = int(value) & 0xFF


def _docc_round(x):
    t = np.trunc(x)
    return t + np.copysign(type(x)(1), x) if abs(x - t) >= 0.5 else t
)";
}

std::string PythonLanguageExtension::allocation(const types::IType& pointer_type, const std::string& bytes) {
    auto pointer = dynamic_cast<const types::Pointer*>(&pointer_type);
    if (!pointer) {
        unsupported("allocating " + pointer_type.print());
    }
    std::vector<std::string> inner_dims;
    types::PrimitiveType element = types::PrimitiveType::UInt8;
    if (pointer->has_pointee_type()) {
        const types::IType* current = &pointer->pointee_type();
        while (auto array = dynamic_cast<const types::Array*>(current)) {
            inner_dims.push_back(this->expression(array->num_elements()));
            current = &array->element_type();
        }
        auto scalar = dynamic_cast<const types::Scalar*>(current);
        if (!scalar) {
            unsupported("allocating " + pointer_type.print());
        }
        element = scalar->primitive_type();
    }
    std::string buffer = "np.zeros(_docc_idiv(" + bytes + ", " + std::to_string(byte_width(element)) + "), " +
                         primitive_type(element) + ")";
    if (!inner_dims.empty()) {
        buffer += ".reshape(-1, " + join(inner_dims, ", ") + ")";
    }
    return buffer;
}

std::string PythonLanguageExtension::primitive_type(const types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Void:
            return "None";
        case types::PrimitiveType::Bool:
            return "np.bool_";
        case types::PrimitiveType::Int8:
            return "np.int8";
        case types::PrimitiveType::Int16:
            return "np.int16";
        case types::PrimitiveType::Int32:
            return "np.int32";
        case types::PrimitiveType::Int64:
            return "np.int64";
        case types::PrimitiveType::UInt8:
            return "np.uint8";
        case types::PrimitiveType::UInt16:
            return "np.uint16";
        case types::PrimitiveType::UInt32:
            return "np.uint32";
        case types::PrimitiveType::UInt64:
            return "np.uint64";
        case types::PrimitiveType::Half:
            return "np.float16";
        case types::PrimitiveType::BFloat:
            return "ml_dtypes.bfloat16";
        case types::PrimitiveType::Float:
            return "np.float32";
        case types::PrimitiveType::Double:
            return "np.float64";
        case types::PrimitiveType::X86_FP80:
            return "np.longdouble";
        case types::PrimitiveType::Int128:
        case types::PrimitiveType::UInt128:
        case types::PrimitiveType::FP128:
        case types::PrimitiveType::PPC_FP128:
            unsupported(std::string("primitive type ") + types::primitive_type_to_string(prim_type));
    }
    throw std::runtime_error("Unknown primitive type");
}

std::string PythonLanguageExtension::
    declaration(const std::string& name, const types::IType& type, bool use_initializer, bool use_alignment) {
    return name;
}

std::string PythonLanguageExtension::type_cast(const std::string& name, const types::IType& type) {
    if (auto scalar = dynamic_cast<const types::Scalar*>(&type)) {
        return primitive_type(scalar->primitive_type()) + "(" + name + ")";
    }
    auto pointer = dynamic_cast<const types::Pointer*>(&type);
    if (!pointer) {
        unsupported("casting " + name + " to " + type.print());
    }
    if (!pointer->has_pointee_type() || (this->function_.exists(name) && this->function_.type(name) == type)) {
        return name;
    }

    // Reinterpret the buffer: NumPy views the same bytes with the pointee's dtype and inner shape.
    std::vector<std::string> inner_dims;
    const types::IType* element = &pointer->pointee_type();
    while (auto array = dynamic_cast<const types::Array*>(element)) {
        inner_dims.push_back(this->expression(array->num_elements()));
        element = &array->element_type();
    }
    auto scalar = dynamic_cast<const types::Scalar*>(element);
    if (!scalar) {
        unsupported("reinterpreting " + name + " as " + type.print());
    }
    std::string view = name + ".view(" + primitive_type(scalar->primitive_type()) + ")";
    if (!inner_dims.empty()) {
        view += ".reshape(-1, " + join(inner_dims, ", ") + ")";
    }
    return view;
}

std::string PythonLanguageExtension::subset(const types::IType& type, const data_flow::Subset& sub) {
    // Indices into one contiguous allocation form a single NumPy index tuple; each pointer starts a new one.
    std::string result;
    std::vector<std::string> group;
    const types::IType* current = &type;
    for (auto& index : sub) {
        if (auto pointer = dynamic_cast<const types::Pointer*>(current)) {
            if (!pointer->has_pointee_type()) {
                unsupported("indexing an opaque pointer");
            }
            if (!group.empty()) {
                result += "[" + join(group, ", ") + "]";
                group.clear();
            }
            group.push_back(this->expression(index));
            current = &pointer->pointee_type();
        } else if (auto array = dynamic_cast<const types::Array*>(current)) {
            group.push_back(this->expression(index));
            current = &array->element_type();
        } else if (dynamic_cast<const types::Scalar*>(current)) {
            break;
        } else {
            unsupported("subsets of " + current->print());
        }
    }
    if (!group.empty()) {
        result += "[" + join(group, ", ") + "]";
    }
    return result;
}

std::string PythonLanguageExtension::expression(const symbolic::Expression expr) {
    PythonSymbolicPrinter printer(this->function_);
    return printer.apply(expr);
}

std::string PythonLanguageExtension::literal(const std::string& data, const types::IType& type) {
    if (symbolic::is_nullptr(data)) {
        return "None";
    }
    auto scalar = dynamic_cast<const types::Scalar*>(&type);
    if (!scalar) {
        unsupported("constant " + data + " of type " + type.print());
    }
    auto prim = scalar->primitive_type();
    if (prim == types::PrimitiveType::Bool) {
        return std::string("np.bool_(") + (integer_bits(data) != 0 ? "True" : "False") + ")";
    }
    if (is_integral(prim)) {
        return primitive_type(prim) + "(" + integer_value(integer_bits(data), prim) + ")";
    }
    return primitive_type(prim) + "(" + floating_value(data) + ")";
}

std::string PythonLanguageExtension::access_node(const data_flow::AccessNode& node) {
    if (auto constant = dynamic_cast<const data_flow::ConstantNode*>(&node)) {
        return literal(constant->data(), constant->type());
    }
    if (this->function_.is_external(node.data())) {
        unsupported("external container " + node.data());
    }
    return node.data();
}

std::string PythonLanguageExtension::identifier(const std::string& name) const {
    static const std::unordered_set<std::string> reserved = {
        "False", "None",     "True",  "and",    "as",    "assert", "async",  "await",    "break",
        "class", "continue", "def",   "del",    "elif",  "else",   "except", "finally",  "for",
        "from",  "global",   "if",    "import", "in",    "is",     "lambda", "nonlocal", "not",
        "or",    "pass",     "raise", "return", "try",   "while",  "with",   "yield",    "np",
        "math",  "range",    "type",  "int",    "float", "bool",   "len",    "min",      "max",
    };
    return reserved.count(name) ? name + "_" : name;
}

std::string PythonLanguageExtension::loop_schedule(const structured_control_flow::ScheduleType& schedule) const {
    if (schedule.category() == structured_control_flow::ScheduleTypeCategory::Vectorizer ||
        structured_control_flow::ScheduleType_Serializable::is_set(schedule)) {
        return structured_control_flow::ScheduleType_Sequential::value();
    }
    return schedule.value();
}

std::string PythonLanguageExtension::tasklet(const data_flow::Tasklet& tasklet) {
    std::vector<std::string> in;
    for (auto& name : tasklet.inputs()) {
        in.push_back(identifier(name));
    }
    auto binary = [&](const std::string& op) {
        return in.at(0) + " " + op + " " + in.at(1);
    };
    auto unordered = [&](const std::string& op) {
        return "np.isnan(" + in.at(0) + ") or np.isnan(" + in.at(1) + ") or " + binary(op);
    };
    auto three_way = [&]() {
        return "(-1 if " + binary("<") + " else (1 if " + binary(">") + " else 0))";
    };

    switch (tasklet.code()) {
        case data_flow::TaskletCode::assign:
            return in.at(0);
        case data_flow::TaskletCode::fp_neg:
            return "-" + in.at(0);
        case data_flow::TaskletCode::fp_add:
        case data_flow::TaskletCode::int_add:
            return binary("+");
        case data_flow::TaskletCode::fp_sub:
        case data_flow::TaskletCode::int_sub:
            return binary("-");
        case data_flow::TaskletCode::fp_mul:
        case data_flow::TaskletCode::int_mul:
            return binary("*");
        case data_flow::TaskletCode::fp_div:
            return binary("/");
        case data_flow::TaskletCode::fp_rem:
            return "np.fmod(" + in.at(0) + ", " + in.at(1) + ")";
        case data_flow::TaskletCode::fp_fma:
            return in.at(0) + " * " + in.at(1) + " + " + in.at(2);
        case data_flow::TaskletCode::fp_oeq:
        case data_flow::TaskletCode::int_eq:
            return binary("==");
        case data_flow::TaskletCode::fp_one:
            return "not np.isnan(" + in.at(0) + ") and not np.isnan(" + in.at(1) + ") and " + binary("!=");
        case data_flow::TaskletCode::int_ne:
            return binary("!=");
        case data_flow::TaskletCode::fp_ogt:
        case data_flow::TaskletCode::int_sgt:
        case data_flow::TaskletCode::int_ugt:
            return binary(">");
        case data_flow::TaskletCode::fp_oge:
        case data_flow::TaskletCode::int_sge:
        case data_flow::TaskletCode::int_uge:
            return binary(">=");
        case data_flow::TaskletCode::fp_olt:
        case data_flow::TaskletCode::int_slt:
        case data_flow::TaskletCode::int_ult:
            return binary("<");
        case data_flow::TaskletCode::fp_ole:
        case data_flow::TaskletCode::int_sle:
        case data_flow::TaskletCode::int_ule:
            return binary("<=");
        case data_flow::TaskletCode::fp_ord:
            return "not (np.isnan(" + in.at(0) + ") or np.isnan(" + in.at(1) + "))";
        case data_flow::TaskletCode::fp_uno:
            return "np.isnan(" + in.at(0) + ") or np.isnan(" + in.at(1) + ")";
        case data_flow::TaskletCode::fp_ueq:
            return unordered("==");
        case data_flow::TaskletCode::fp_une:
            return unordered("!=");
        case data_flow::TaskletCode::fp_ugt:
            return unordered(">");
        case data_flow::TaskletCode::fp_uge:
            return unordered(">=");
        case data_flow::TaskletCode::fp_ult:
            return unordered("<");
        case data_flow::TaskletCode::fp_ule:
            return unordered("<=");
        case data_flow::TaskletCode::int_sdiv:
            return "_docc_sdiv(" + in.at(0) + ", " + in.at(1) + ")";
        case data_flow::TaskletCode::int_srem:
            return "_docc_srem(" + in.at(0) + ", " + in.at(1) + ")";
        // Unsigned operands are non-negative, so floor and truncating division agree.
        case data_flow::TaskletCode::int_udiv:
            return binary("//");
        case data_flow::TaskletCode::int_urem:
            return binary("%");
        case data_flow::TaskletCode::int_and:
            return binary("&");
        case data_flow::TaskletCode::int_or:
            return binary("|");
        case data_flow::TaskletCode::int_xor:
            return binary("^");
        case data_flow::TaskletCode::int_shl:
            return binary("<<");
        case data_flow::TaskletCode::int_lshr:
        case data_flow::TaskletCode::int_ashr:
            return binary(">>");
        case data_flow::TaskletCode::int_smin:
        case data_flow::TaskletCode::int_umin:
            return "min(" + in.at(0) + ", " + in.at(1) + ")";
        case data_flow::TaskletCode::int_smax:
        case data_flow::TaskletCode::int_umax:
            return "max(" + in.at(0) + ", " + in.at(1) + ")";
        case data_flow::TaskletCode::int_scmp:
        case data_flow::TaskletCode::int_ucmp:
            return three_way();
        case data_flow::TaskletCode::int_abs:
            return "abs(" + in.at(0) + ")";
    }
    throw std::invalid_argument("Invalid tasklet code");
}

std::string PythonLanguageExtension::zero(const types::PrimitiveType prim_type) {
    return primitive_type(prim_type) + "(0)";
}

std::string PythonLanguageExtension::comment(const std::string& text) {
    return "# " + text;
}

std::string PythonLanguageExtension::assignment(const std::string& lhs, const std::string& rhs) {
    return lhs + " = " + rhs;
}

std::string PythonLanguageExtension::
    converting_assignment(const std::string& lhs, const types::Scalar& type, const std::string& rhs) {
    return lhs + " = _docc_cast(" + primitive_type(type.primitive_type()) + ", " + rhs + ")";
}

std::string PythonLanguageExtension::
    local_declaration(const std::string& name, const types::IType& type, const std::string& init) {
    if (init.empty()) {
        return "";
    }
    if (auto scalar = dynamic_cast<const types::Scalar*>(&type)) {
        return name + " = " + primitive_type(scalar->primitive_type()) + "(" + init + ")";
    }
    return name + " = " + init;
}

std::string PythonLanguageExtension::copy_bytes(const std::string& dst, const std::string& src, const std::string& size) {
    unsupported("copying arrays or structures by value");
}

std::string PythonLanguageExtension::break_statement() {
    return "break";
}

std::string PythonLanguageExtension::continue_statement() {
    for (auto it = open_bodies_.rbegin(); it != open_bodies_.rend(); ++it) {
        if (it->is_loop) {
            // A for loop emitted as while must still advance its induction variable.
            return it->loop_update.empty() ? "continue" : it->loop_update + "; continue";
        }
    }
    return "continue";
}

std::string PythonLanguageExtension::return_statement(const std::string& value) {
    return "return " + value;
}

void PythonLanguageExtension::deallocation(PrettyPrinter& stream, const std::string& container, const types::IType& type) {
}

std::string PythonLanguageExtension::null_literal() {
    return "None";
}

std::string PythonLanguageExtension::address_of(const std::string& expr) {
    unsupported("taking the address of " + expr);
}

std::string PythonLanguageExtension::
    element_address(const std::string& base, const types::IType& base_type, const data_flow::Subset& indices) {
    auto pointer = dynamic_cast<const types::Pointer*>(&base_type);
    if (pointer && indices.size() == 1 && pointer->has_pointee_type() &&
        dynamic_cast<const types::Scalar*>(&pointer->pointee_type())) {
        return base + "[" + expression(indices.at(0)) + ":]";
    }
    return address_of(base + subset(base_type, indices));
}

std::string PythonLanguageExtension::dereference(const std::string& expr) {
    unsupported("dereferencing " + expr);
}

std::string PythonLanguageExtension::size_of(const std::string& expr) {
    unsupported("sizeof " + expr);
}

std::string PythonLanguageExtension::reinterpret_cast_expr(const std::string& expr, const std::string& type) {
    unsupported("calling through function pointer " + expr);
}

Body PythonLanguageExtension::
    begin_for(PrettyPrinter& stream, const structured_control_flow::StructuredLoop& loop, const LoopHints& hints) {
    auto indvar = loop.indvar()->get_name();
    auto stride = loop.stride();
    if (!stride.is_null() && stride->as_int() != 0) {
        auto bound = stride->as_int() > 0 ? loop.canonical_bound_upper() : loop.canonical_bound_lower();
        if (!bound.is_null()) {
            stream << "for " << indvar << " in range(" << this->expression(loop.init()) << ", "
                   << this->expression(bound);
            if (stride->as_int() != 1) {
                stream << ", " << stride->as_int();
            }
            stream << "):" << std::endl;
            return open(stream, Body::Kind::Block, true);
        }
    }

    stream << assignment(indvar, this->expression(loop.init())) << std::endl;
    stream << "while " << this->expression(loop.condition()) << ":" << std::endl;
    return open(stream, Body::Kind::Block, true, assignment(indvar, this->expression(loop.update())));
}

Body PythonLanguageExtension::begin_while(PrettyPrinter& stream) {
    stream << "while True:" << std::endl;
    return open(stream, Body::Kind::Block, true);
}

Body PythonLanguageExtension::
    begin_branch(PrettyPrinter& stream, size_t index, size_t count, const symbolic::Condition& condition) {
    stream << (index == 0 ? "if " : "elif ") << this->expression(condition) << ":" << std::endl;
    return open(stream, Body::Kind::Block, false);
}

Body PythonLanguageExtension::begin_scope(PrettyPrinter& stream) {
    open_bodies_.push_back({Body::Kind::Scope, false, ""});
    return Body{Body::Kind::Scope, stream.content()};
}

Body PythonLanguageExtension::open(PrettyPrinter& stream, Body::Kind kind, bool is_loop, const std::string& loop_update) {
    open_bodies_.push_back({kind, is_loop, loop_update});
    stream.changeIndent(+4);
    return Body{kind, stream.content()};
}

void PythonLanguageExtension::end(PrettyPrinter& stream, const Body& body) {
    if (open_bodies_.empty() || open_bodies_.back().kind != body.kind) {
        throw std::logic_error("Python codegen: bodies closed out of order");
    }
    auto closing = open_bodies_.back();
    open_bodies_.pop_back();
    if (body.kind == Body::Kind::Scope) {
        return;
    }
    if (!closing.loop_update.empty()) {
        stream << closing.loop_update << std::endl;
    } else if (stream.content() == body.content_mark) {
        stream << "pass" << std::endl;
    }
    stream.changeIndent(-4);
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Infty& x) {
    str_ = x.is_negative_infinity() ? "-math.inf" : "math.inf";
}

void PythonSymbolicPrinter::bvisit(const SymEngine::BooleanAtom& x) {
    str_ = x.get_val() ? "True" : "False";
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Symbol& x) {
    const std::string& name = x.get_name();
    if (symbolic::is_nullptr(name)) {
        unsupported("null pointers in symbolic expressions");
    }
    if (this->function_.is_external(name)) {
        unsupported("external symbol " + name);
    }
    if (this->function_.exists(name) && this->function_.type(name).type_id() == types::TypeID::Pointer) {
        unsupported("pointer " + name + " in symbolic expressions");
    }
    str_ = name;
}

void PythonSymbolicPrinter::bvisit(const SymEngine::And& x) {
    std::vector<std::string> parts;
    for (auto& arg : x.get_container()) {
        parts.push_back(apply(arg));
    }
    str_ = parenthesize(join(parts, " and "));
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Or& x) {
    std::vector<std::string> parts;
    for (auto& arg : x.get_container()) {
        parts.push_back(apply(arg));
    }
    str_ = parenthesize(join(parts, " or "));
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Not& x) {
    str_ = parenthesize("not " + apply(x.get_arg()));
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Equality& x) {
    str_ = parenthesize(apply(x.get_args()[0]) + " == " + apply(x.get_args()[1]));
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Unequality& x) {
    str_ = parenthesize(apply(x.get_args()[0]) + " != " + apply(x.get_args()[1]));
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Min& x) {
    std::vector<std::string> parts;
    for (auto& arg : x.get_args()) {
        parts.push_back(apply(arg));
    }
    str_ = parts.size() == 1 ? parts.front() : "min(" + join(parts, ", ") + ")";
}

void PythonSymbolicPrinter::bvisit(const SymEngine::Max& x) {
    std::vector<std::string> parts;
    for (auto& arg : x.get_args()) {
        parts.push_back(apply(arg));
    }
    str_ = parts.size() == 1 ? parts.front() : "max(" + join(parts, ", ") + ")";
}

void PythonSymbolicPrinter::bvisit(const SymEngine::FunctionSymbol& x) {
    auto args = x.get_args();
    const std::string& name = x.get_name();
    if (name == "idiv") {
        str_ = "_docc_idiv(" + apply(args[0]) + ", " + apply(args[1]) + ")";
    } else if (name == "imod") {
        str_ = "_docc_imod(" + apply(args[0]) + ", " + apply(args[1]) + ")";
    } else if (name == "iabs") {
        str_ = "abs(" + apply(args[0]) + ")";
    } else if (name == "zext_i64") {
        str_ = "_docc_zext_i64(" + apply(args[0]) + ")";
    } else if (name == "trunc_i32") {
        str_ = "_docc_trunc_i32(" + apply(args[0]) + ")";
    } else if (name == "bit_xor") {
        str_ = "((" + apply(args[0]) + ") ^ (" + apply(args[1]) + "))";
    } else if (name == "sizeof") {
        auto& type = dynamic_cast<const symbolic::SizeOfTypeFunction&>(x).get_type();
        auto scalar = dynamic_cast<const types::Scalar*>(&type);
        if (!scalar) {
            unsupported("sizeof(" + type.print() + ")");
        }
        str_ = std::to_string(byte_width(scalar->primitive_type()));
    } else {
        unsupported("function symbol " + name);
    }
}

void PythonSymbolicPrinter::_print_pow(
    std::ostringstream& o,
    const SymEngine::RCP<const SymEngine::Basic>& a,
    const SymEngine::RCP<const SymEngine::Basic>& b
) {
    if (SymEngine::eq(*a, *SymEngine::E)) {
        o << "math.exp(" << apply(b) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 2))) {
        o << "math.sqrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::rational(1, 3))) {
        o << "math.cbrt(" << apply(a) << ")";
    } else if (SymEngine::eq(*b, *SymEngine::integer(2))) {
        o << "((" << apply(a) << ") * (" << apply(a) << "))";
    } else {
        o << "((" << apply(a) << ") ** (" << apply(b) << "))";
    }
}

} // namespace codegen
} // namespace sdfg
