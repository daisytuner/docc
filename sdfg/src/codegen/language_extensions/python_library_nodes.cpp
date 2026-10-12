#include "sdfg/codegen/language_extensions/python_library_nodes.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/language_extensions/python_language_extension.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/calloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"

namespace sdfg {
namespace codegen {

namespace {

PythonLanguageExtension& python(LanguageExtension& language_extension) {
    return dynamic_cast<PythonLanguageExtension&>(language_extension);
}

/// Malloc and Calloc: the output pointer becomes a zero-filled NumPy buffer.
class AllocationDispatcher : public LibraryNodeDispatcher {
    symbolic::Expression bytes_;

public:
    AllocationDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node,
        symbolic::Expression bytes
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), bytes_(bytes) {
    }

    void dispatch_code_with_edges(
        CodegenOutput& out, std::vector<DispatchInput>& inputs, std::vector<DispatchOutput>& outputs
    ) override {
        auto& oedge = *data_flow_graph_.out_edges(node_).begin();
        auto& lang = python(language_extension_);
        out.stream << node_.output(0) << " = " << lang.allocation(oedge.base_type(), lang.expression(bytes_))
                   << std::endl;
        register_output(outputs.at(0), node_.output(0));
    }
};

class FreeDispatcher : public LibraryNodeDispatcher {
public:
    using LibraryNodeDispatcher::LibraryNodeDispatcher;

    void dispatch_code_with_edges(
        CodegenOutput& out, std::vector<DispatchInput>& inputs, std::vector<DispatchOutput>& outputs
    ) override {
    }
};

/// Memcpy, Memmove and Memset operate on the raw bytes behind their pointer inputs.
class ByteOperationDispatcher : public LibraryNodeDispatcher {
    std::string helper_;
    std::vector<symbolic::Expression> arguments_;

public:
    ByteOperationDispatcher(
        LanguageExtension& language_extension,
        const Function& function,
        const data_flow::DataFlowGraph& data_flow_graph,
        const data_flow::LibraryNode& node,
        std::string helper,
        std::vector<symbolic::Expression> arguments
    )
        : LibraryNodeDispatcher(language_extension, function, data_flow_graph, node), helper_(std::move(helper)),
          arguments_(std::move(arguments)) {
    }

    void dispatch_code_with_edges(
        CodegenOutput& out, std::vector<DispatchInput>& inputs, std::vector<DispatchOutput>& outputs
    ) override {
        out.stream << helper_ << "(";
        for (auto& input : inputs) {
            out.stream << input.expr << ", ";
        }
        for (size_t i = 0; i < arguments_.size(); ++i) {
            out.stream << (i > 0 ? ", " : "") << language_extension_.expression(arguments_[i]);
        }
        out.stream << ")" << std::endl;
    }
};

std::string cmath_call(math::cmath::CMathFunction function, const std::vector<std::string>& in) {
    using math::cmath::CMathFunction;
    auto call = [&](const std::string& callee) {
        std::string result = callee + "(";
        for (size_t i = 0; i < in.size(); ++i) {
            result += (i > 0 ? ", " : "") + in[i];
        }
        return result + ")";
    };
    // Python's math module computes in double precision; cast back to the operand's type.
    auto via_math = [&](const std::string& callee) {
        return "type(" + in.at(0) + ")(" + call(callee) + ")";
    };

    switch (function) {
        case CMathFunction::sin:
        case CMathFunction::cos:
        case CMathFunction::tan:
        case CMathFunction::sinh:
        case CMathFunction::cosh:
        case CMathFunction::tanh:
        case CMathFunction::exp:
        case CMathFunction::exp2:
        case CMathFunction::expm1:
        case CMathFunction::log:
        case CMathFunction::log10:
        case CMathFunction::log2:
        case CMathFunction::log1p:
        case CMathFunction::sqrt:
        case CMathFunction::cbrt:
        case CMathFunction::hypot:
        case CMathFunction::fabs:
        case CMathFunction::ceil:
        case CMathFunction::floor:
        case CMathFunction::trunc:
        case CMathFunction::rint:
        case CMathFunction::fmod:
        case CMathFunction::ldexp:
        case CMathFunction::nextafter:
        case CMathFunction::copysign:
        case CMathFunction::fmax:
        case CMathFunction::fmin:
            return call(std::string("np.") + math::cmath::cmath_function_to_stem(function));
        case CMathFunction::asin:
            return call("np.arcsin");
        case CMathFunction::acos:
            return call("np.arccos");
        case CMathFunction::atan:
            return call("np.arctan");
        case CMathFunction::atan2:
            return call("np.arctan2");
        case CMathFunction::asinh:
            return call("np.arcsinh");
        case CMathFunction::acosh:
            return call("np.arccosh");
        case CMathFunction::atanh:
            return call("np.arctanh");
        case CMathFunction::exp10:
            return "np.power(type(" + in.at(0) + ")(10), " + in.at(0) + ")";
        case CMathFunction::pow:
            return call("np.power");
        case CMathFunction::erf:
            return via_math("math.erf");
        case CMathFunction::erfc:
            return via_math("math.erfc");
        case CMathFunction::tgamma:
            return via_math("math.gamma");
        case CMathFunction::lgamma:
            return via_math("math.lgamma");
        case CMathFunction::remainder:
            return via_math("math.remainder");
        case CMathFunction::round:
            return call("_docc_round");
        case CMathFunction::roundeven:
        case CMathFunction::nearbyint:
            return call("np.rint");
        case CMathFunction::lround:
        case CMathFunction::llround:
            return "np.int64(" + call("_docc_round") + ")";
        case CMathFunction::lrint:
        case CMathFunction::llrint:
            return "np.int64(" + call("np.rint") + ")";
        case CMathFunction::scalbn:
        case CMathFunction::scalbln:
            return call("np.ldexp");
        case CMathFunction::nexttoward:
            return "np.nextafter(" + in.at(0) + ", type(" + in.at(0) + ")(" + in.at(1) + "))";
        case CMathFunction::fdim:
            return "np.maximum(" + in.at(0) + " - " + in.at(1) + ", type(" + in.at(0) + ")(0))";
        case CMathFunction::fma:
            return in.at(0) + " * " + in.at(1) + " + " + in.at(2);
        case CMathFunction::ilogb:
            return "np.int32(np.frexp(" + in.at(0) + ")[1] - 1)";
        case CMathFunction::logb:
            return "type(" + in.at(0) + ")(np.frexp(" + in.at(0) + ")[1] - 1)";
        case CMathFunction::frexp:
        case CMathFunction::modf:
            break;
    }
    throw std::runtime_error(
        std::string("Python codegen: CMath ") + math::cmath::cmath_function_to_stem(function) + " is not supported"
    );
}

class CMathDispatcher : public LibraryNodeDispatcher {
public:
    using LibraryNodeDispatcher::LibraryNodeDispatcher;

    void dispatch_code(
        PrettyPrinter& stream, PrettyPrinter& globals_stream, CodeSnippetFactory& library_snippet_factory
    ) override {
        auto& node = static_cast<const math::cmath::CMathNode&>(node_);
        stream << node.outputs().at(0) << " = " << cmath_call(node.function(), node.inputs()) << std::endl;
    }
};

template<typename Factory>
void register_python(const data_flow::LibraryNodeCode& code, Factory factory) {
    LibraryNodeDispatcherRegistry::instance().register_library_node_dispatcher(
        code,
        ImplementationType_Python,
        [factory](
            LanguageExtension& language_extension,
            const Function& function,
            const data_flow::DataFlowGraph& data_flow_graph,
            const data_flow::LibraryNode& node
        ) -> std::unique_ptr<LibraryNodeDispatcher> {
            return factory(language_extension, function, data_flow_graph, node);
        }
    );
}

} // namespace

void register_python_library_node_dispatchers() {
    register_python(stdlib::LibraryNodeType_Malloc, [](auto& lang, auto& function, auto& graph, auto& node) {
        auto& malloc = static_cast<const stdlib::MallocNode&>(node);
        return std::make_unique<AllocationDispatcher>(lang, function, graph, node, malloc.size());
    });
    register_python(stdlib::LibraryNodeType_Calloc, [](auto& lang, auto& function, auto& graph, auto& node) {
        auto& calloc = static_cast<const stdlib::CallocNode&>(node);
        return std::make_unique<
            AllocationDispatcher>(lang, function, graph, node, symbolic::mul(calloc.num(), calloc.size()));
    });
    register_python(stdlib::LibraryNodeType_Free, [](auto& lang, auto& function, auto& graph, auto& node) {
        return std::make_unique<FreeDispatcher>(lang, function, graph, node);
    });
    register_python(stdlib::LibraryNodeType_Memcpy, [](auto& lang, auto& function, auto& graph, auto& node) {
        auto& memcpy = static_cast<const stdlib::MemcpyNode&>(node);
        return std::make_unique<ByteOperationDispatcher>(
            lang, function, graph, node, "_docc_memcpy", std::vector<symbolic::Expression>{memcpy.count()}
        );
    });
    register_python(stdlib::LibraryNodeType_Memmove, [](auto& lang, auto& function, auto& graph, auto& node) {
        auto& memmove = static_cast<const stdlib::MemmoveNode&>(node);
        return std::make_unique<ByteOperationDispatcher>(
            lang, function, graph, node, "_docc_memcpy", std::vector<symbolic::Expression>{memmove.count()}
        );
    });
    register_python(stdlib::LibraryNodeType_Memset, [](auto& lang, auto& function, auto& graph, auto& node) {
        auto& memset = static_cast<const stdlib::MemsetNode&>(node);
        return std::make_unique<ByteOperationDispatcher>(
            lang, function, graph, node, "_docc_memset", std::vector<symbolic::Expression>{memset.value(), memset.num()}
        );
    });
    register_python(math::cmath::LibraryNodeType_CMath, [](auto& lang, auto& function, auto& graph, auto& node) {
        return std::make_unique<CMathDispatcher>(lang, function, graph, node);
    });
}

} // namespace codegen
} // namespace sdfg
