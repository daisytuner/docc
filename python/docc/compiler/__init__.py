from docc.compiler.compiled_sdfg import (
    CompiledSDFG,
    DoccPerformanceWarning,
    warn_device_residency_failed,
    warn_host_to_device_copies,
)
from docc.compiler.docc_program import DoccProgram, DoccOptions

try:
    from importlib.metadata import version

    __version__ = version("docc-compiler")
except Exception as e:
    __version__ = "dev"
