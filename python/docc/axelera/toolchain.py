"""Compiles generated AxScript modules with Axelera's alto compiler.

The Axelera SDK pins its own TVM fork and Python version, so alto runs in a separate interpreter given by
DOCC_AXELERA_PYTHON instead of docc's environment.
"""

import os
import subprocess

_BUILD_SCRIPT = """
import importlib.util
import sys
from pathlib import Path

from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
from axelera.compiler.alto.compiler.driver import compile as alto_compile

module_path, name, output_dir = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("_docc_axscript", module_path)
module = importlib.util.module_from_spec(spec)
# The TVMScript parser looks the module up to read the kernel's source.
sys.modules[spec.name] = module
spec.loader.exec_module(module)
config = AltoConfig(
    output_dir=Path(output_dir),
    generation=HardwareGeneration.OMEGA,
    target=Target.AICORE_SIM,
    num_cores=module._docc_num_cores,
    enable_tiling_injection=True,
    unpack_axm_archive=True,
    clean_output_dir=True,
)
alto_compile(getattr(module, name), config)
"""


def build(module_path: str, name: str) -> str:
    """Compile the ir_module ``name`` of ``module_path`` for Metis; returns alto's output directory."""
    python = os.environ.get("DOCC_AXELERA_PYTHON")
    if not python:
        raise RuntimeError(
            "The axscript codegen compiles with Axelera's alto: set DOCC_AXELERA_PYTHON to a Python "
            "interpreter that has axelera-compiler installed"
        )
    output_dir = os.path.join(os.path.dirname(module_path), "alto")
    # The SDK's helper executables (e.g. axkernelcc) are installed next to its interpreter.
    path = os.path.dirname(python) + os.pathsep + os.environ.get("PATH", "")
    result = subprocess.run(
        [python, "-c", _BUILD_SCRIPT, module_path, name, output_dir],
        capture_output=True,
        text=True,
        env=dict(os.environ, PATH=path),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"alto failed to compile {module_path}:\n{result.stderr[-4000:]}"
        )
    return output_dir
