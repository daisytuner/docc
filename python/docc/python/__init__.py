from docc.python.ast_parser import ASTParser
from docc.python.ast_utils import get_debug_info, get_unique_id
from docc.python.functions.scipy import SciPyHandler
from docc.python.functions.numpy import NumPyHandler
from docc.python.python_program import PythonProgram, native, _map_python_type
from docc.compiler.target_registry import (
    register_target,
    register_target_overrides,
    unregister_target,
    reset_target_registry,
)

# Backward compatibility alias - ExpressionVisitor is now merged into ASTParser
ExpressionVisitor = ASTParser
