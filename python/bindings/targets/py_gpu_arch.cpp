#include "py_gpu_arch.h"
#include <sdfg/targets/rocm/rocm_arch.h>

void register_gpu_arch(py::module& m, sdfg::plugins::Context& context) {
    // RocmArch: a ROCm GPU architecture descriptor (e.g. gfx1201, gfx90a).
    py::class_<sdfg::gpu::rocm::RocmArch>(m, "RocmArch")
        .def_static(
            "get_from_name",
            [](const std::string& name) -> const sdfg::gpu::rocm::RocmArch& {
                auto* arch = sdfg::gpu::rocm::rocm_arch_parse(name);
                if (!arch) {
                    throw std::runtime_error("Unknown ROCm architecture: " + name);
                }
                return *arch;
            },
            py::arg("name"),
            py::return_value_policy::reference,
            "Return the RocmArch matching the given gfx name (e.g. 'gfx1201'); raises if unknown."
        )
        .def_static(
            "get_current",
            []() -> const sdfg::gpu::rocm::RocmArch& {
                const char* env = std::getenv("DOCC_ROCM_ARCH");
                if (!env || std::string(env).empty()) {
                    throw std::runtime_error("DOCC_ROCM_ARCH is not set");
                }
                auto* arch = sdfg::gpu::rocm::rocm_arch_parse(env);
                if (!arch) {
                    throw std::runtime_error(std::string("Unknown ROCm architecture in DOCC_ROCM_ARCH: ") + env);
                }
                return *arch;
            },
            py::return_value_policy::reference,
            "Return the RocmArch named by the DOCC_ROCM_ARCH environment variable; raises if unset/unknown."
        )
        .def_static(
            "current_name",
            []() -> py::object {
                const char* env = std::getenv("DOCC_ROCM_ARCH");
                if (!env || std::string(env).empty()) {
                    return py::none();
                }
                return py::cast(std::string(env));
            },
            "Return the value of DOCC_ROCM_ARCH, or None if it is unset."
        )
        .def_property_readonly("name", [](const sdfg::gpu::rocm::RocmArch& self) { return self.name(); })
        .def("__repr__", [](const sdfg::gpu::rocm::RocmArch& self) { return "<RocmArch '" + self.name() + "'>"; });
}
