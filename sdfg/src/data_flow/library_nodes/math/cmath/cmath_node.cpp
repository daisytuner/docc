#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/symbolic/symbolic.h"

#include <string>
#include <unordered_map>

#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace cmath {

CMathFunction string_to_cmath_function(const std::string& name) {
    static const std::unordered_map<std::string, CMathFunction> function_map = {
        {"sin", CMathFunction::sin},
        {"sinf", CMathFunction::sin},
        {"sinl", CMathFunction::sin},
        {"cos", CMathFunction::cos},
        {"cosf", CMathFunction::cos},
        {"cosl", CMathFunction::cos},
        {"tan", CMathFunction::tan},
        {"tanf", CMathFunction::tan},
        {"tanl", CMathFunction::tan},
        {"asin", CMathFunction::asin},
        {"asinf", CMathFunction::asin},
        {"asinl", CMathFunction::asin},
        {"acos", CMathFunction::acos},
        {"acosf", CMathFunction::acos},
        {"acosl", CMathFunction::acos},
        {"atan", CMathFunction::atan},
        {"atanf", CMathFunction::atan},
        {"atanl", CMathFunction::atan},
        {"atan2", CMathFunction::atan2},
        {"atan2f", CMathFunction::atan2},
        {"atan2l", CMathFunction::atan2},
        {"sinh", CMathFunction::sinh},
        {"sinhf", CMathFunction::sinh},
        {"sinhl", CMathFunction::sinh},
        {"cosh", CMathFunction::cosh},
        {"coshf", CMathFunction::cosh},
        {"coshl", CMathFunction::cosh},
        {"tanh", CMathFunction::tanh},
        {"tanhf", CMathFunction::tanh},
        {"tanhl", CMathFunction::tanh},
        {"asinh", CMathFunction::asinh},
        {"asinhf", CMathFunction::asinh},
        {"asinhl", CMathFunction::asinh},
        {"acosh", CMathFunction::acosh},
        {"acoshf", CMathFunction::acosh},
        {"acoshl", CMathFunction::acosh},
        {"atanh", CMathFunction::atanh},
        {"atanhf", CMathFunction::atanh},
        {"atanhl", CMathFunction::atanh},
        {"exp", CMathFunction::exp},
        {"expf", CMathFunction::exp},
        {"expl", CMathFunction::exp},
        {"exp2", CMathFunction::exp2},
        {"exp2f", CMathFunction::exp2},
        {"exp2l", CMathFunction::exp2},
        {"exp10", CMathFunction::exp10},
        {"exp10f", CMathFunction::exp10},
        {"exp10l", CMathFunction::exp10},
        {"expm1", CMathFunction::expm1},
        {"expm1f", CMathFunction::expm1},
        {"expm1l", CMathFunction::expm1},
        {"log", CMathFunction::log},
        {"logf", CMathFunction::log},
        {"logl", CMathFunction::log},
        {"log10", CMathFunction::log10},
        {"log10f", CMathFunction::log10},
        {"log10l", CMathFunction::log10},
        {"log2", CMathFunction::log2},
        {"log2f", CMathFunction::log2},
        {"log2l", CMathFunction::log2},
        {"log1p", CMathFunction::log1p},
        {"log1pf", CMathFunction::log1p},
        {"log1pl", CMathFunction::log1p},
        {"pow", CMathFunction::pow},
        {"powf", CMathFunction::pow},
        {"powl", CMathFunction::pow},
        {"sqrt", CMathFunction::sqrt},
        {"sqrtf", CMathFunction::sqrt},
        {"sqrtl", CMathFunction::sqrt},
        {"cbrt", CMathFunction::cbrt},
        {"cbrtf", CMathFunction::cbrt},
        {"cbrtl", CMathFunction::cbrt},
        {"hypot", CMathFunction::hypot},
        {"hypotf", CMathFunction::hypot},
        {"hypotl", CMathFunction::hypot},
        {"erf", CMathFunction::erf},
        {"erff", CMathFunction::erf},
        {"erfl", CMathFunction::erf},
        {"erfc", CMathFunction::erfc},
        {"erfcf", CMathFunction::erfc},
        {"erfcl", CMathFunction::erfc},
        {"tgamma", CMathFunction::tgamma},
        {"tgammaf", CMathFunction::tgamma},
        {"tgammal", CMathFunction::tgamma},
        {"lgamma", CMathFunction::lgamma},
        {"lgammaf", CMathFunction::lgamma},
        {"lgammal", CMathFunction::lgamma},
        {"fabs", CMathFunction::fabs},
        {"fabsf", CMathFunction::fabs},
        {"fabsl", CMathFunction::fabs},
        {"ceil", CMathFunction::ceil},
        {"ceilf", CMathFunction::ceil},
        {"ceill", CMathFunction::ceil},
        {"floor", CMathFunction::floor},
        {"floorf", CMathFunction::floor},
        {"floorl", CMathFunction::floor},
        {"trunc", CMathFunction::trunc},
        {"truncf", CMathFunction::trunc},
        {"truncl", CMathFunction::trunc},
        {"round", CMathFunction::round},
        {"roundf", CMathFunction::round},
        {"roundl", CMathFunction::round},
        {"lround", CMathFunction::lround},
        {"lroundf", CMathFunction::lround},
        {"lroundl", CMathFunction::lround},
        {"llround", CMathFunction::llround},
        {"llroundf", CMathFunction::llround},
        {"llroundl", CMathFunction::llround},
        {"roundeven", CMathFunction::roundeven},
        {"roundevenf", CMathFunction::roundeven},
        {"roundevenl", CMathFunction::roundeven},
        {"nearbyint", CMathFunction::nearbyint},
        {"nearbyintf", CMathFunction::nearbyint},
        {"nearbyintl", CMathFunction::nearbyint},
        {"rint", CMathFunction::rint},
        {"rintf", CMathFunction::rint},
        {"rintl", CMathFunction::rint},
        {"lrint", CMathFunction::lrint},
        {"lrintf", CMathFunction::lrint},
        {"lrintl", CMathFunction::lrint},
        {"llrint", CMathFunction::llrint},
        {"llrintf", CMathFunction::llrint},
        {"llrintl", CMathFunction::llrint},
        {"fmod", CMathFunction::fmod},
        {"fmodf", CMathFunction::fmod},
        {"fmodl", CMathFunction::fmod},
        {"remainder", CMathFunction::remainder},
        {"remainderf", CMathFunction::remainder},
        {"remainderl", CMathFunction::remainder},
        {"frexp", CMathFunction::frexp},
        {"frexpf", CMathFunction::frexp},
        {"frexpl", CMathFunction::frexp},
        {"ldexp", CMathFunction::ldexp},
        {"ldexpf", CMathFunction::ldexp},
        {"ldexpl", CMathFunction::ldexp},
        {"modf", CMathFunction::modf},
        {"modff", CMathFunction::modf},
        {"modfl", CMathFunction::modf},
        {"scalbn", CMathFunction::scalbn},
        {"scalbnf", CMathFunction::scalbn},
        {"scalbnl", CMathFunction::scalbn},
        {"scalbln", CMathFunction::scalbln},
        {"scalblnf", CMathFunction::scalbln},
        {"scalblnl", CMathFunction::scalbln},
        {"ilogb", CMathFunction::ilogb},
        {"ilogbf", CMathFunction::ilogb},
        {"ilogbl", CMathFunction::ilogb},
        {"logb", CMathFunction::logb},
        {"logbf", CMathFunction::logb},
        {"logbl", CMathFunction::logb},
        {"nextafter", CMathFunction::nextafter},
        {"nextafterf", CMathFunction::nextafter},
        {"nextafterl", CMathFunction::nextafter},
        {"nexttoward", CMathFunction::nexttoward},
        {"nexttowardf", CMathFunction::nexttoward},
        {"nexttowardl", CMathFunction::nexttoward},
        {"copysign", CMathFunction::copysign},
        {"copysignf", CMathFunction::copysign},
        {"copysignl", CMathFunction::copysign},
        {"fmax", CMathFunction::fmax},
        {"fmaxf", CMathFunction::fmax},
        {"fmaxl", CMathFunction::fmax},
        {"fmin", CMathFunction::fmin},
        {"fminf", CMathFunction::fmin},
        {"fminl", CMathFunction::fmin},
        {"fdim", CMathFunction::fdim},
        {"fdimf", CMathFunction::fdim},
        {"fdiml", CMathFunction::fdim},
        {"fma", CMathFunction::fma},
        {"fmaf", CMathFunction::fma},
        {"fmal", CMathFunction::fma},
    };

    auto it = function_map.find(name);
    if (it != function_map.end()) {
        return it->second;
    }

    throw std::runtime_error("Unknown CMath function: " + name);
}

CMathNode::CMathNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    CMathFunction function,
    types::PrimitiveType primitive_type
)
    : MathNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_CMath, {"_out"}, {}, data_flow::ImplementationType_NONE
      ),
      function_(function), primitive_type_(primitive_type) {
    for (size_t i = 0; i < cmath_function_to_arity(function); i++) {
        this->inputs_.push_back("_in" + std::to_string(i + 1));
    }
}

CMathFunction CMathNode::function() const { return this->function_; }

types::PrimitiveType CMathNode::primitive_type() const { return this->primitive_type_; }

std::string CMathNode::name() const { return get_cmath_intrinsic_name(this->function_, this->primitive_type_); }

symbolic::SymbolSet CMathNode::symbols() const { return {}; }

void CMathNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    return;
}

void CMathNode::validate(const Function& function) const {
    MathNode::validate(function);

    if (!types::is_floating_point(this->primitive_type_)) {
        throw InvalidSDFGException("CMathNode: Primitive type must be a floating point type");
    }

    auto& dataflow = this->get_parent();
    if (this->inputs_.size() != dataflow.in_degree(*this)) {
        throw InvalidSDFGException("CMathNode: Mismatch between number of inputs and in-degree of the node");
    }
    if (this->outputs_.size() != dataflow.out_degree(*this)) {
        throw InvalidSDFGException("CMathNode: Mismatch between number of outputs and out-degree of the node");
    }
    for (const auto& iedge : dataflow.in_edges(*this)) {
        auto inferred_type = types::infer_type(function, iedge.base_type(), iedge.subset());
        auto type_id = inferred_type->type_id();
        bool type_ok = false;
        if (type_id == types::TypeID::Scalar) {
            type_ok = true;
        } else if (type_id == types::TypeID::Tensor) {
            auto& tensor = dynamic_cast<types::Tensor&>(*inferred_type);
            if (tensor.is_scalar()) {
                type_ok = true;
            }
        }
        if (!type_ok) {
            throw InvalidSDFGException(
                "CMathNode: Input type on " + std::to_string(this->element_id()) + ":" + iedge.dst_conn() +
                " must be scalar, but was " + std::to_string(static_cast<int>(type_id))
            );
        }
        auto& scalar_type = static_cast<const types::Scalar&>(*inferred_type);
        if (scalar_type.primitive_type() != this->primitive_type_) {
            std::string input_primitive_type_str = types::primitive_type_to_string(scalar_type.primitive_type());
            std::string node_primitive_type_str = types::primitive_type_to_string(this->primitive_type_);
            throw InvalidSDFGException(
                "CMathNode: Input primitive type " + input_primitive_type_str + " does not match node primitive type " +
                node_primitive_type_str + " for function " + cmath_function_to_stem(this->function_)
            );
        }
    }
    for (const auto& oedge : dataflow.out_edges(*this)) {
        auto inferred_type = types::infer_type(function, oedge.base_type(), oedge.subset());
        auto type_id = inferred_type->type_id();
        bool type_ok = false;
        if (type_id == types::TypeID::Scalar) {
            type_ok = true;
        } else if (type_id == types::TypeID::Tensor) {
            auto& tensor = dynamic_cast<types::Tensor&>(*inferred_type);
            if (tensor.is_scalar()) {
                type_ok = true;
            }
        }
        if (!type_ok) {
            throw InvalidSDFGException(
                "Output: Input type in " + std::to_string(this->element_id()) + " must be scalar, but was " +
                std::to_string(static_cast<int>(type_id))
            );
        }
        auto& scalar_type = static_cast<const types::Scalar&>(*inferred_type);
        if (this->function_ == CMathFunction::lrint || this->function_ == CMathFunction::llrint ||
            this->function_ == CMathFunction::lround || this->function_ == CMathFunction::llround) {
            if (!types::is_integer(scalar_type.primitive_type())) {
                std::string output_primitive_type_str = types::primitive_type_to_string(scalar_type.primitive_type());
                throw InvalidSDFGException(
                    "CMathNode: Output primitive type must be an integer type for lrint, llrint, lround, llround "
                    "functions. Found: " +
                    output_primitive_type_str
                );
            }
        } else if (scalar_type.primitive_type() != this->primitive_type_) {
            std::string output_primitive_type_str = types::primitive_type_to_string(scalar_type.primitive_type());
            std::string node_primitive_type_str = types::primitive_type_to_string(this->primitive_type_);
            throw InvalidSDFGException(
                "CMathNode: Output primitive type " + output_primitive_type_str +
                " does not match node primitive type " + node_primitive_type_str
            );
        }
    }
}

std::unique_ptr<data_flow::DataFlowNode> CMathNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        CMathNode>(new CMathNode(element_id, this->debug_info(), vertex, parent, this->function_, this->primitive_type_)
    );
}

symbolic::Expression CMathNode::flop() const { return symbolic::one(); }

std::string CMathNode::toStr() const {
    return LibraryNode::toStr() + "(" + get_cmath_intrinsic_name(this->function_, this->primitive_type_) + ")";
}

data_flow::EdgeRemoveOption CMathNode::
    can_remove_out_edge(const data_flow::DataFlowGraph& graph, const data_flow::Memlet* memlet) const {
    if (graph.out_edges_for_connector(*this, memlet->src_conn()).size() > 1) {
        return data_flow::EdgeRemoveOption::Trivially;
    } else {
        // trick to allow removal of Tasklets, without having general mark-and-sweep logic to checkl tha that all
        // outputs are
        return data_flow::EdgeRemoveOption::RemoveNodeAfter;
    }
}

nlohmann::json CMathNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const CMathNode& node = static_cast<const CMathNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    j["name"] = node.name();
    j["function_stem"] = cmath_function_to_stem(node.function());
    j["primitive_type"] = static_cast<int>(node.primitive_type());

    return j;
}

data_flow::LibraryNode& CMathNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    // Backward compatibility
    if (code != LibraryNodeType_CMath.value() && code != LibraryNodeType_CMath_Deprecated.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    // Try new format first (with function_stem and primitive_type)
    CMathFunction function;
    types::PrimitiveType prim_type;

    if (j.contains("function_stem") && j.contains("primitive_type")) {
        // New format
        auto stem = j["function_stem"].get<std::string>();
        function = string_to_cmath_function(stem);
        prim_type = static_cast<types::PrimitiveType>(j["primitive_type"].get<int>());
    } else {
        // Backward compatibility: old format with just "name"
        auto name = j["name"].get<std::string>();
        function = string_to_cmath_function(name);
        // Infer primitive type from the suffix
        if (name.back() == 'f') {
            prim_type = types::PrimitiveType::Float;
        } else if (name.back() == 'l') {
            prim_type = types::PrimitiveType::X86_FP80; // Assuming long double
        } else {
            prim_type = types::PrimitiveType::Double;
        }
    }

    return builder.add_library_node<CMathNode>(parent, debug_info, function, prim_type);
}

CMathNodeDispatcher::CMathNodeDispatcher(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const CMathNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void CMathNodeDispatcher::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    auto& node = static_cast<const CMathNode&>(this->node_);

    stream << node.outputs().at(0) << " = ";
    stream << node.name() << "(";
    for (size_t i = 0; i < node.inputs().size(); i++) {
        stream << node.inputs().at(i);
        if (i < node.inputs().size() - 1) {
            stream << ", ";
        }
    }
    stream << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace cmath
} // namespace math
} // namespace sdfg
