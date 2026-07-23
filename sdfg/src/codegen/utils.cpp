#include "sdfg/codegen/utils.h"

#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/function.h"

namespace sdfg {
namespace codegen {

// Constructor
PrettyPrinter::PrettyPrinter(int indent, bool frozen)
    : owned_stream(std::make_unique<std::stringstream>()), stream(*owned_stream.get()), indentSize(indent),
      frozen_(frozen) {}

PrettyPrinter::PrettyPrinter(std::ostream& stream, int indent, bool frozen)
    : stream(stream), indentSize(indent), frozen_(frozen) {}

// Set the indentation level
void PrettyPrinter::setIndent(int indent) { indentSize = indent; };

int PrettyPrinter::indent() const { return indentSize; };

int PrettyPrinter::changeIndent(int delta) { return indentSize += delta; };

// Get the underlying string
std::string PrettyPrinter::str() const { return owned_stream->str(); };

// Clear the stringstream content
void PrettyPrinter::clear() {
    owned_stream->str("");
    owned_stream->clear();
}

// Overload for manipulators (like std::endl)
PrettyPrinter& PrettyPrinter::operator<<(std::ostream& (*manip)(std::ostream&) ) {
    if (frozen_) {
        throw std::runtime_error("PrettyPrinter is frozen");
    }
    stream << manip;
    // Reset indent application on new lines
    if (manip == static_cast<std::ostream& (*) (std::ostream&)>(std::endl)) {
        isNewLine = true;
    }
    return *this;
};

// Apply indentation only at the beginning of a new line
void PrettyPrinter::applyIndent() {
    if (isNewLine && indentSize > 0) {
        stream << std::setw(indentSize) << "";
        isNewLine = false;
    }
};

Reference::Reference(const types::IType& reference_) : reference_(reference_.clone()) {};

Reference::Reference(
    types::StorageType storage_type, size_t alignment, const std::string& initializer, const types::IType& reference_
)
    : IType(storage_type, alignment, initializer), reference_(reference_.clone()) {};

std::unique_ptr<types::IType> Reference::clone() const {
    return std::make_unique<Reference>(this->storage_type(), this->alignment(), this->initializer(), *this->reference_);
};

types::TypeID Reference::type_id() const { return types::TypeID::Reference; };

types::PrimitiveType Reference::primitive_type() const { return this->reference_->primitive_type(); };

bool Reference::is_symbol() const { return false; };

const types::IType& Reference::reference_type() const { return *this->reference_; };

bool Reference::operator==(const types::IType& other) const {
    if (auto reference = dynamic_cast<const Reference*>(&other)) {
        return *(this->reference_) == *reference->reference_ && this->alignment_ == reference->alignment_;
    } else {
        return false;
    }
};

std::string Reference::print() const { return "Reference(" + this->reference_->print() + ")"; };


std::string complex_type_name(types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::CHalf:
            return "__daisy_type_complex_half";
        case types::PrimitiveType::CBFloat:
            return "__daisy_type_complex_bfloat";
        case types::PrimitiveType::CFloat:
            return "__daisy_type_complex_float";
        case types::PrimitiveType::CDouble:
            return "__daisy_type_complex_double";
        case types::PrimitiveType::CFP128:
            return "__daisy_type_complex_fp128";
        default:
            throw InvalidSDFGException("complex_type_name: not a complex primitive type");
    }
};

namespace {

// Real "compute type" used to evaluate a complex operation component-wise. Narrow element types
// (half/bfloat) have limited native arithmetic support, so their components are widened to float
// for the computation and narrowed back on assignment.
std::string complex_compute_type(types::PrimitiveType prim) {
    switch (prim) {
        case types::PrimitiveType::CDouble:
            return "double";
        case types::PrimitiveType::CFP128:
            return "__float128";
        default:
            return "float";
    }
}

} // namespace

std::string complex_computation(const data_flow::Tasklet& tasklet, const Function& function) {
    if (!data_flow::is_complex(tasklet.code())) {
        throw InvalidSDFGException("complex_computation: tasklet is not a complex operation");
    }

    // The operand type (always complex) drives the component compute type.
    auto& graph = tasklet.get_parent();
    types::PrimitiveType operand_prim = types::PrimitiveType::CFloat;
    for (auto& iedge : graph.in_edges(tasklet)) {
        operand_prim = iedge.result_type(function)->primitive_type();
        break;
    }
    const std::string re = "(" + complex_compute_type(operand_prim) + ")";

    const auto& inputs = tasklet.inputs();
    const std::string& o = tasklet.output();
    const std::string& a = inputs.at(0);

    std::stringstream out;
    switch (tasklet.code()) {
        case data_flow::TaskletCode::complex_add: {
            const std::string& b = inputs.at(1);
            out << o << ".x = " << re << a << ".x + " << re << b << ".x; ";
            out << o << ".y = " << re << a << ".y + " << re << b << ".y;";
            break;
        }
        case data_flow::TaskletCode::complex_sub: {
            const std::string& b = inputs.at(1);
            out << o << ".x = " << re << a << ".x - " << re << b << ".x; ";
            out << o << ".y = " << re << a << ".y - " << re << b << ".y;";
            break;
        }
        case data_flow::TaskletCode::complex_mul: {
            const std::string& b = inputs.at(1);
            out << o << ".x = " << re << a << ".x * " << re << b << ".x - " << re << a << ".y * " << re << b << ".y; ";
            out << o << ".y = " << re << a << ".x * " << re << b << ".y + " << re << a << ".y * " << re << b << ".x;";
            break;
        }
        case data_flow::TaskletCode::complex_div: {
            const std::string& b = inputs.at(1);
            const std::string denom = "(" + re + b + ".x * " + re + b + ".x + " + re + b + ".y * " + re + b + ".y)";
            out << o << ".x = (" << re << a << ".x * " << re << b << ".x + " << re << a << ".y * " << re << b
                << ".y) / " << denom << "; ";
            out << o << ".y = (" << re << a << ".y * " << re << b << ".x - " << re << a << ".x * " << re << b
                << ".y) / " << denom << ";";
            break;
        }
        case data_flow::TaskletCode::complex_neg:
            out << o << ".x = -" << re << a << ".x; ";
            out << o << ".y = -" << re << a << ".y;";
            break;
        case data_flow::TaskletCode::complex_real:
            out << o << " = " << a << ".x;";
            break;
        case data_flow::TaskletCode::complex_imag:
            out << o << " = " << a << ".y;";
            break;
        case data_flow::TaskletCode::complex_eq: {
            const std::string& b = inputs.at(1);
            out << o << " = (" << re << a << ".x == " << re << b << ".x && " << re << a << ".y == " << re << b
                << ".y);";
            break;
        }
        case data_flow::TaskletCode::complex_ne: {
            const std::string& b = inputs.at(1);
            out << o << " = !(" << re << a << ".x == " << re << b << ".x && " << re << a << ".y == " << re << b
                << ".y);";
            break;
        }
        default:
            throw InvalidSDFGException("complex_computation: unhandled complex tasklet code");
    }
    return out.str();
};

std::string complex_support_preamble(bool device) {
    // Element type of the half-precision component. GPU toolchains expose __fp16, matching the
    // scalar Half mapping of the CUDA/ROCm language extensions.
    const std::string half_elem = device ? "__fp16" : "_Float16";

    std::stringstream out;
    out << "/* Complex type support (generated by sdfglib) */" << std::endl;

    // Dedicated 2-component vector types with `.x` (real) / `.y` (imaginary) members. Named with a
    // reserved prefix so they never collide with a toolchain's native float2/double2 definitions.
    out << "typedef struct { float x; float y; } __daisy_type_complex_float;" << std::endl;
    out << "typedef struct { double x; double y; } __daisy_type_complex_double;" << std::endl;
    out << "typedef struct { " << half_elem << " x; " << half_elem << " y; } __daisy_type_complex_half;" << std::endl;
    out << "typedef struct { __bf16 x; __bf16 y; } __daisy_type_complex_bfloat;" << std::endl;
    out << "typedef struct { __float128 x; __float128 y; } __daisy_type_complex_fp128;" << std::endl;

    return out.str();
};


} // namespace codegen
} // namespace sdfg
