#pragma once

namespace sdfg {
namespace codegen {

/// Registers the NumPy implementations of the stdlib and CMath library nodes.
void register_python_library_node_dispatchers();

} // namespace codegen
} // namespace sdfg
