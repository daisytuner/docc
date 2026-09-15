#include <pybind11/pybind11.h>
#include <sdfg/plugins/plugins.h>

namespace py = pybind11;

// Register bindings that model GPU (CUDA, ROCM) architecture descriptions (chip and generation specific things, like
// shader model, operation support, cache sizes etc.)
void register_gpu_arch(py::module& m, sdfg::plugins::Context& context);
