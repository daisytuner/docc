#include "py_gpu_arch.h"
#include <sdfg/targets/cuda/cuda_arch.h>
#include <sdfg/targets/gpu/gpu_mma_fragment.h>
#include <sdfg/targets/rocm/rocm_arch.h>

void register_gpu_arch(py::module& m, sdfg::plugins::Context& context) {
    // MmaBlockSize: an MxNxK tensor-core block size (e.g. 32x32x8).
    py::class_<sdfg::gpu::MmaBlockSize>(m, "MmaBlockSize")
        .def(
            py::init([](int block_m, int block_n, int block_k) {
                return sdfg::gpu::MmaBlockSize{block_m, block_n, block_k};
            }),
            py::arg("m"),
            py::arg("n"),
            py::arg("k"),
            "Create an MMA block size from its M, N and K dimensions."
        )
        .def_static(
            "parse",
            &sdfg::gpu::MmaBlockSize::parse_block_size,
            py::arg("block_size"),
            "Parse an MxNxK block size string (e.g. '32x32x8')."
        )
        .def_readwrite("m", &sdfg::gpu::MmaBlockSize::m)
        .def_readwrite("n", &sdfg::gpu::MmaBlockSize::n)
        .def_readwrite("k", &sdfg::gpu::MmaBlockSize::k)
        .def("__repr__", [](const sdfg::gpu::MmaBlockSize& self) {
            return "<MmaBlockSize '" + self.toStr() + "'>";
        });

    py::class_<sdfg::gpu::GpuArch>(m, "GpuArch")
        .def_property_readonly("name", &sdfg::gpu::GpuArch::name, "Get the GPU architecture name")
        .def_static(
            "get_from_schedule_type",
            [](const sdfg::structured_control_flow::ScheduleType& schedule) -> const sdfg::gpu::GpuArch* {
                return sdfg::gpu::GpuArch::get_from_schedule_type(schedule);
            },
            py::arg("schedule"),
            py::return_value_policy::reference,
            "Return the GpuArch matching the given ScheduleType; returns None if unknown."
        );

    // RocmArch: a ROCm GPU architecture descriptor (e.g. gfx1201, gfx90a).
    py::class_<sdfg::gpu::rocm::RocmArch, sdfg::gpu::GpuArch>(m, "RocmArch")
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
                if (auto from_env = sdfg::gpu::rocm::rocm_arch_from_env()) {
                    return *from_env;
                } else if (auto from_hw = sdfg::gpu::rocm::rocm_arch_from_available_hardware()) {
                    return *from_hw;
                }
                throw std::runtime_error("No DOCC_ROCM_ARCH set or locally found ROCM hardware");
            },
            py::return_value_policy::reference,
            "Return the RocmArch named by the DOCC_ROCM_ARCH environment variable; raises if unset/unknown."
        )
        .def_static(
            "current_name",
            []() -> py::object {
                const sdfg::gpu::GpuArch* arch = nullptr;
                if (auto from_env = sdfg::gpu::rocm::rocm_arch_from_env()) {
                    arch = from_env;
                } else if (auto from_hw = sdfg::gpu::rocm::rocm_arch_from_available_hardware()) {
                    arch = from_hw;
                }
                return arch ? py::cast(arch->name()) : py::none();
            },
            "Return the value of DOCC_ROCM_ARCH, or None if it is unset."
        )
        .def_property_readonly(
            "name",
            [](const sdfg::gpu::rocm::RocmArch& self) {
                return self.name();
            }
        )
        .def("__repr__", [](const sdfg::gpu::rocm::RocmArch& self) {
            return "<RocmArch '" + self.name() + "'>";
        });

    // CudaArch: a CUDA GPU architecture descriptor (e.g. sm_70, sm_120).
    py::class_<sdfg::gpu::cuda::CudaArch, sdfg::gpu::GpuArch>(m, "CudaArch")
        .def_static(
            "get_from_name",
            [](const std::string& name) -> const sdfg::gpu::cuda::CudaArch& {
                auto* arch = sdfg::gpu::cuda::cuda_arch_parse(name);
                if (!arch) {
                    throw std::runtime_error("Unknown CUDA architecture: " + name);
                }
                return *arch;
            },
            py::arg("name"),
            py::return_value_policy::reference,
            "Return the CudaArch matching the given gfx name (e.g. 'sm_120'); raises if unknown."
        )
        .def_static(
            "get_current",
            []() -> const sdfg::gpu::cuda::CudaArch& {
                if (auto from_env = sdfg::gpu::cuda::cuda_arch_from_env()) {
                    return *from_env;
                } else if (auto from_hw = sdfg::gpu::cuda::cuda_arch_from_available_hardware()) {
                    return *from_hw;
                }
                throw std::runtime_error("No DOCC_CUDA_ARCH set or locally found CUDA hardware");
            },
            py::return_value_policy::reference,
            "Return the CudaArch named by the DOCC_CUDA_ARCH environment variable. If unset, looks at available "
            "hardware; raises if unset/unknown."
        )
        .def_static(
            "current_name",
            []() -> py::object {
                const sdfg::gpu::GpuArch* arch = nullptr;
                if (auto from_env = sdfg::gpu::cuda::cuda_arch_from_env()) {
                    arch = from_env;
                } else if (auto from_hw = sdfg::gpu::cuda::cuda_arch_from_available_hardware()) {
                    arch = from_hw;
                }
                return arch ? py::cast(arch->name()) : py::none();
            },
            "Return the value of DOCC_RCZDA_ARCH, or None if it is unset."
        )
        .def_property_readonly(
            "name",
            [](const sdfg::gpu::cuda::CudaArch& self) {
                return self.name();
            }
        )
        .def("__repr__", [](const sdfg::gpu::cuda::CudaArch& self) {
            return "<CudaArch '" + self.name() + "'>";
        });
}
