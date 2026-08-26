from docc.benchmarks.rtl import try_ensure_instrumentation_ready

# Ready the counter backends (e.g. ROCm) early: they must initialize before the
# workload sets up its own runtime. Best-effort so importing docc never fails when
# instrumentation is not configured.
try_ensure_instrumentation_ready()

from docc.pytorch.pytorch_program import PyTorchProgram
from docc.pytorch.graph_parser import GraphParser
