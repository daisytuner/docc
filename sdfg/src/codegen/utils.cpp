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
            return "half2";
        case types::PrimitiveType::CBFloat:
            return "bfloat162";
        case types::PrimitiveType::CFloat:
            return "float2";
        case types::PrimitiveType::CDouble:
            return "double2";
        case types::PrimitiveType::CFP128:
            return "fp128_2";
        default:
            throw InvalidSDFGException("complex_type_name: not a complex primitive type");
    }
};

std::string complex_op_suffix(types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::CHalf:
            return "h";
        case types::PrimitiveType::CBFloat:
            return "bf";
        case types::PrimitiveType::CFloat:
            return "f";
        case types::PrimitiveType::CDouble:
            return "d";
        case types::PrimitiveType::CFP128:
            return "q";
        default:
            throw InvalidSDFGException("complex_op_suffix: not a complex primitive type");
    }
};

std::string complex_tasklet(sdfg::Function& function, const data_flow::Tasklet& tasklet) {
    // Determine the element suffix from the (complex) operand type.
    auto& graph = tasklet.get_parent();
    auto in_edges = graph.in_edges(tasklet);
    if (in_edges.begin() == in_edges.end()) {
        throw InvalidSDFGException("complex_tasklet: tasklet has no inputs");
    }
    auto operand_type = (*in_edges.begin()).result_type(function);
    std::string suffix = complex_op_suffix(operand_type->primitive_type());

    auto& inputs = tasklet.inputs();
    switch (tasklet.code()) {
        case data_flow::TaskletCode::complex_neg:
            return "__daisy_cneg_" + suffix + "(" + inputs.at(0) + ")";
        case data_flow::TaskletCode::complex_real:
            return "__daisy_creal_" + suffix + "(" + inputs.at(0) + ")";
        case data_flow::TaskletCode::complex_imag:
            return "__daisy_cimag_" + suffix + "(" + inputs.at(0) + ")";
        case data_flow::TaskletCode::complex_add:
            return "__daisy_cadd_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        case data_flow::TaskletCode::complex_sub:
            return "__daisy_csub_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        case data_flow::TaskletCode::complex_mul:
            return "__daisy_cmul_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        case data_flow::TaskletCode::complex_div:
            return "__daisy_cdiv_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        case data_flow::TaskletCode::complex_eq:
            return "__daisy_ceq_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        case data_flow::TaskletCode::complex_ne:
            return "__daisy_cne_" + suffix + "(" + inputs.at(0) + ", " + inputs.at(1) + ")";
        default:
            throw InvalidSDFGException("complex_tasklet: not a complex tasklet code");
    }
};

std::string complex_support_preamble(bool device) {
    const std::string qual = device ? "__host__ __device__ static inline" : "static inline";

    std::stringstream out;
    out << "/* Complex type support (generated by sdfglib) */" << std::endl;

    // Component vector types. On GPU, float2/double2/half2/__nv_bfloat162 are provided by the
    // toolchain and reused; on CPU they are defined here. fp128_2 has no native type anywhere.
    if (device) {
        out << "typedef __nv_bfloat162 bfloat162;" << std::endl;
    } else {
        out << "typedef struct { float x; float y; } float2;" << std::endl;
        out << "typedef struct { double x; double y; } double2;" << std::endl;
        out << "typedef struct { _Float16 x; _Float16 y; } half2;" << std::endl;
        out << "typedef struct { __bf16 x; __bf16 y; } bfloat162;" << std::endl;
    }
    out << "typedef struct { __float128 x; __float128 y; } fp128_2;" << std::endl;

    // Element-wise helper functions. Arithmetic is carried out in a wide type (WT) so that
    // narrow element types (half/bfloat) that lack native arithmetic are computed via float.
    out << "#define __DAISY_DEFINE_COMPLEX(CT, ST, WT, SUF, QUAL) \\" << std::endl;
    out << "  QUAL CT __daisy_cadd_##SUF(CT a, CT b) { CT r; r.x = (ST)((WT)a.x + (WT)b.x); r.y = "
           "(ST)((WT)a.y + (WT)b.y); return r; } \\"
        << std::endl;
    out << "  QUAL CT __daisy_csub_##SUF(CT a, CT b) { CT r; r.x = (ST)((WT)a.x - (WT)b.x); r.y = "
           "(ST)((WT)a.y - (WT)b.y); return r; } \\"
        << std::endl;
    out << "  QUAL CT __daisy_cmul_##SUF(CT a, CT b) { CT r; r.x = (ST)((WT)a.x * (WT)b.x - (WT)a.y * (WT)b.y); "
           "r.y = (ST)((WT)a.x * (WT)b.y + (WT)a.y * (WT)b.x); return r; } \\"
        << std::endl;
    out << "  QUAL CT __daisy_cdiv_##SUF(CT a, CT b) { CT r; WT d = (WT)b.x * (WT)b.x + (WT)b.y * (WT)b.y; "
           "r.x = (ST)(((WT)a.x * (WT)b.x + (WT)a.y * (WT)b.y) / d); "
           "r.y = (ST)(((WT)a.y * (WT)b.x - (WT)a.x * (WT)b.y) / d); return r; } \\"
        << std::endl;
    out << "  QUAL CT __daisy_cneg_##SUF(CT a) { CT r; r.x = (ST)(-(WT)a.x); r.y = (ST)(-(WT)a.y); return r; } \\"
        << std::endl;
    out << "  QUAL ST __daisy_creal_##SUF(CT a) { return a.x; } \\" << std::endl;
    out << "  QUAL ST __daisy_cimag_##SUF(CT a) { return a.y; } \\" << std::endl;
    out << "  QUAL bool __daisy_ceq_##SUF(CT a, CT b) { return (WT)a.x == (WT)b.x && (WT)a.y == (WT)b.y; } \\"
        << std::endl;
    out << "  QUAL bool __daisy_cne_##SUF(CT a, CT b) { return !((WT)a.x == (WT)b.x && (WT)a.y == (WT)b.y); }"
        << std::endl;

    if (device) {
        out << "__DAISY_DEFINE_COMPLEX(float2, float, float, f, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(double2, double, double, d, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(half2, __half, float, h, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(bfloat162, __nv_bfloat16, float, bf, " << qual << ")" << std::endl;
        // __float128 is a host-only type; keep fp128 helpers off the device.
        out << "__DAISY_DEFINE_COMPLEX(fp128_2, __float128, __float128, q, __host__ inline)" << std::endl;
    } else {
        out << "__DAISY_DEFINE_COMPLEX(float2, float, float, f, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(double2, double, double, d, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(half2, _Float16, float, h, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(bfloat162, __bf16, float, bf, " << qual << ")" << std::endl;
        out << "__DAISY_DEFINE_COMPLEX(fp128_2, __float128, __float128, q, " << qual << ")" << std::endl;
    }
    out << "#undef __DAISY_DEFINE_COMPLEX" << std::endl;

    return out.str();
};


} // namespace codegen
} // namespace sdfg
