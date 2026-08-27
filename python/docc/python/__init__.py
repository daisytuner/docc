from docc.python.ast_parser import ASTParser
from docc.python.ast_utils import get_debug_info, get_unique_id
from docc.python.functions.numpy import NumPyHandler
from docc.python.python_program import PythonProgram, native, _map_python_type
from docc.compiler.target_registry import (
    register_target,
    register_target_overrides,
    unregister_target,
    reset_target_registry,
)
from docc.benchmarks.rtl import try_ensure_instrumentation_ready

# Backward compatibility alias - ExpressionVisitor is now merged into ASTParser
ExpressionVisitor = ASTParser

# Ready the counter backends (e.g. ROCm) early: they must initialize before the
# workload sets up its own runtime. Best-effort so importing docc never fails when
# instrumentation is not configured.
try_ensure_instrumentation_ready()
