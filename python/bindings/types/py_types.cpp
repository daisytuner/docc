#include "py_types.h"

#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/array.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/structure.h>
#include <sdfg/types/tensor.h>
#include <sdfg/types/type.h>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace sdfg::types;

void register_types(py::module& m) {
    // PrimitiveType
    py::enum_<PrimitiveType>(m, "PrimitiveType")
        .value("Void", PrimitiveType::Void)
        .value("Bool", PrimitiveType::Bool)
        .value("Int8", PrimitiveType::Int8)
        .value("Int16", PrimitiveType::Int16)
        .value("Int32", PrimitiveType::Int32)
        .value("Int64", PrimitiveType::Int64)
        .value("Int128", PrimitiveType::Int128)
        .value("UInt8", PrimitiveType::UInt8)
        .value("UInt16", PrimitiveType::UInt16)
        .value("UInt32", PrimitiveType::UInt32)
        .value("UInt64", PrimitiveType::UInt64)
        .value("UInt128", PrimitiveType::UInt128)
        .value("Half", PrimitiveType::Half)
        .value("BFloat", PrimitiveType::BFloat)
        .value("Float", PrimitiveType::Float)
        .value("Double", PrimitiveType::Double)
        .value("X86_FP80", PrimitiveType::X86_FP80)
        .value("FP128", PrimitiveType::FP128)
        .value("PPC_FP128", PrimitiveType::PPC_FP128)
        .export_values();

    // StorageType
    py::class_<StorageType>(m, "StorageType")
        .def(py::init<const std::string&>(), py::arg("value"))
        .def(
            py::init([](const std::string& value, const std::string& allocation_size) {
                return new StorageType(
                    value,
                    sdfg::symbolic::parse(allocation_size),
                    StorageType::AllocationType::Unmanaged,
                    StorageType::AllocationType::Unmanaged
                );
            }),
            py::arg("value"),
            py::arg("allocation_size")
        )
        .def_static("CPU_Stack", []() { return StorageType::CPU_Stack(); })
        .def_static("CPU_Heap", []() { return StorageType::CPU_Heap(); })
        .def_static("NV_Generic", []() { return StorageType::NV_Generic(); })
        .def_static("NV_Global", []() { return StorageType::NV_Global(); })
        .def_static("NV_Shared", []() { return StorageType::NV_Shared(); })
        .def_static("NV_Constant", []() { return StorageType::NV_Constant(); })
        .def_static("NV_Symbol", []() { return StorageType::NV_Symbol(); })
        .def_static("AMD_Generic", []() { return StorageType("AMD_Generic"); })
        .def_property_readonly("value", [](const StorageType& st) { return st.value(); })
        .def("__repr__", [](const StorageType& st) { return "<StorageType value='" + st.value() + "'>"; });

    // IType
    py::class_<IType>(m, "Type")
        .def("print", &IType::print)
        .def("__repr__", &IType::print)
        .def_property_readonly("primitive_type", &IType::primitive_type);

    // Scalar
    py::class_<Scalar, IType>(m, "Scalar").def(py::init<PrimitiveType>(), py::arg("primitive_type"));

    // Array
    py::class_<Array, IType>(m, "Array")
        .def(
            py::init([](const IType& element_type, const std::string& num_elements) {
                return new Array(element_type, sdfg::symbolic::parse(num_elements));
            }),
            py::arg("element_type"),
            py::arg("num_elements")
        )
        .def_property_readonly("element_type", &Array::element_type)
        .def_property_readonly("num_elements", [](const Array& self) { return self.num_elements()->__str__(); });

    // Pointer
    py::class_<Pointer, IType>(m, "Pointer")
        .def(py::init<>())
        .def(py::init<const IType&>(), py::arg("pointee_type"))
        .def(
            py::init([](const IType& pointee_type, const StorageType& storage_type) {
                return new Pointer(storage_type, 0, "", pointee_type);
            }),
            py::arg("pointee_type"),
            py::arg("storage_type")
        )
        .def_property_readonly("pointee_type", &Pointer::pointee_type)
        .def("has_pointee_type", &Pointer::has_pointee_type);

    // Structure
    py::class_<Structure, IType>(m, "Structure")
        .def(py::init<const std::string&>(), py::arg("name"))
        .def_property_readonly("name", &Structure::name);

    // Tensor
    py::class_<Tensor, IType>(m, "Tensor")
        .def(
            py::init([](const Scalar& element_type, const std::vector<std::string>& shape) {
                sdfg::symbolic::MultiExpression shape_expr;
                for (const auto& s : shape) {
                    shape_expr.push_back(sdfg::symbolic::parse(s));
                }
                return new Tensor(element_type, shape_expr);
            }),
            py::arg("element_type"),
            py::arg("shape")
        )
        .def(
            py::init([](const Scalar& element_type,
                        const std::vector<std::string>& shape,
                        const std::vector<std::string>& strides,
                        const std::string& offset) {
                sdfg::symbolic::MultiExpression shape_expr;
                for (const auto& s : shape) {
                    shape_expr.push_back(sdfg::symbolic::parse(s));
                }
                sdfg::symbolic::MultiExpression strides_expr;
                for (const auto& s : strides) {
                    strides_expr.push_back(sdfg::symbolic::parse(s));
                }
                return new Tensor(element_type, shape_expr, strides_expr, sdfg::symbolic::parse(offset));
            }),
            py::arg("element_type"),
            py::arg("shape"),
            py::arg("strides"),
            py::arg("offset") = "0"
        )
        .def_property_readonly("element_type", &Tensor::element_type)
        .def_property_readonly(
            "shape",
            [](const Tensor& self) {
                std::vector<std::string> result;
                for (const auto& s : self.shape()) {
                    result.push_back(s->__str__());
                }
                return result;
            }
        )
        .def_property_readonly(
            "strides",
            [](const Tensor& self) {
                std::vector<std::string> result;
                for (const auto& s : self.strides()) {
                    result.push_back(s->__str__());
                }
                return result;
            }
        )
        .def_property_readonly("offset", [](const Tensor& self) { return self.offset()->__str__(); })
        .def("total_elements", [](const Tensor& self) { return self.total_elements()->__str__(); })
        .def("total_size", [](const Tensor& self) { return self.total_size()->__str__(); })
        .def(
            "newaxis", [](const Tensor& self, size_t axis) { return self.newaxis(axis); }, py::arg("axis")
        )
        .def(
            "flip", [](const Tensor& self, size_t axis) { return self.flip(axis); }, py::arg("axis")
        )
        .def(
            "unsqueeze", [](const Tensor& self, size_t axis) { return self.unsqueeze(axis); }, py::arg("axis")
        )
        .def(
            "squeeze", [](const Tensor& self, size_t axis) { return self.squeeze(axis); }, py::arg("axis")
        )
        .def("squeeze", [](const Tensor& self) { return self.squeeze(); })
        .def(
            "reshape",
            [](const Tensor& self, const std::vector<std::string>& new_shape) {
                sdfg::symbolic::MultiExpression shape_expr;
                for (const auto& s : new_shape) {
                    shape_expr.push_back(sdfg::symbolic::parse(s));
                }
                return self.reshape(shape_expr);
            },
            py::arg("new_shape")
        )
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("is_tight", &Tensor::is_tight);
}
