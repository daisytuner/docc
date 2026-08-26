from docc.benchmarks.rtl import try_ensure_instrumentation_ready

# Ready the counter backends (e.g. ROCm) early: they must initialize before the
# workload sets up its own runtime. Best-effort so importing docc never fails when
# instrumentation is not configured.
try_ensure_instrumentation_ready()

from docc.torch.torch_program import (
    TorchProgram,
    compile_torch,
)
from docc.compiler.target_registry import (
    register_target,
    register_target_overrides,
    unregister_target,
)
