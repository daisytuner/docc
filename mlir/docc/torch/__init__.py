import os
import getpass
import hashlib
import shutil
from typing import Any, Optional

from docc.sdfg import StructuredSDFG
from docc.mlir import MLIRModule
from docc.compiler import DoccProgram, CompiledSDFG
from docc.python.target_registry import get_target_schedule_fn, get_target_compile_fn

# Global RPC context for scheduling SDFGs
sdfg_rpc_context = None


class TorchProgram(DoccProgram):
    def __init__(
        self,
        model,
        example_input: Any = None,
        target: str = "none",
        category: str = "server",
        remote_tuning: bool = False,
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
        name: Optional[str] = None,
    ):
        # Determine name from model
        if name is None:
            if hasattr(model, "__class__"):
                name = model.__class__.__name__
            else:
                name = "torch_model"

        # Sanitize name: remove characters that are invalid in filesystem paths
        import re as _re

        name = _re.sub(r"[<>:\"/\\|?*]", "_", name)

        super().__init__(
            name=name,
            target=target,
            category=category,
            remote_tuning=remote_tuning,
            instrumentation_mode=instrumentation_mode,
            capture_args=capture_args,
        )

        self.model = model
        self.example_input = example_input
        self._sdfg: Optional[StructuredSDFG] = None
        self._compiled: Optional[CompiledSDFG] = None
        self._input_info: list = []  # [(shape, dtype), ...] for each input
        self._output_info: list = []  # [(shape, dtype), ...] for each output

    def __call__(self, *args: Any) -> Any:
        import torch

        # Detect input type (torch or numpy)
        is_torch_input = any(isinstance(arg, torch.Tensor) for arg in args)

        # Compile if necessary
        if self._compiled is None:
            self._compiled = self.compile()

        # Convert inputs to numpy
        numpy_args = self._convert_inputs(args)

        # Execute
        result = self._compiled(*numpy_args)

        # get return shape from metadata
        return_shape_str = self._compiled.sdfg.metadata("return_shape")

        # parse shape string back to tuple
        return_shape = tuple(
            int(dim) for dim in return_shape_str.strip("[]").split(",") if dim
        )
        self._output_info = [{"shape": return_shape}]

        # Convert outputs back to torch if inputs were torch
        if is_torch_input:
            result = self._convert_outputs(result, args, return_shape)

        return result

    def compile(
        self,
        output_folder: Optional[str] = None,
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
    ) -> CompiledSDFG:
        original_output_folder = output_folder

        # Resolve options
        if instrumentation_mode is None:
            instrumentation_mode = self.instrumentation_mode or ""
        if capture_args is None:
            capture_args = self.capture_args or False

        # Determine example input
        if self.example_input is None:
            raise ValueError(
                "No example input provided. Either provide example_input during "
                "initialization or pass example inputs to compile()."
            )

        # Generate cache key
        cache_key = self._get_cache_key(self.example_input)

        if original_output_folder is None and cache_key in self.cache:
            return self.cache[cache_key]

        # Determine output folder
        if output_folder is None:
            # Include model-specific code in hash to distinguish different models
            # that share the same class name and input shapes.
            # - FX GraphModules (from torch.compile): use the FX graph code
            # - Regular nn.Modules (from compile_torch): use forward() source
            model_code = ""
            try:
                import torch.fx

                if isinstance(self.model, torch.fx.GraphModule):
                    model_code = self.model.graph.python_code("self").src
            except Exception:
                pass
            if not model_code:
                try:
                    import inspect

                    model_code = inspect.getsource(type(self.model).forward)
                except Exception:
                    pass

            hash_input = f"{self.name}|{self.target}|{self.category}|{cache_key}|{model_code}".encode(
                "utf-8"
            )
            stable_id = hashlib.sha256(hash_input).hexdigest()[:16]

            docc_tmp = os.environ.get("DOCC_TMP")
            if docc_tmp:
                output_folder = f"{docc_tmp}/{self.name}-{stable_id}"
            else:
                user = os.getenv("USER")
                if not user:
                    user = getpass.getuser()
                output_folder = f"/tmp/{user}/DOCC/{self.name}-{stable_id}"

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        # Populate input info from example input
        import torch

        self._input_info = []
        example_inputs = (
            self.example_input
            if isinstance(self.example_input, tuple)
            else (self.example_input,)
        )
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor):
                self._input_info.append(
                    {
                        "shape": tuple(inp.shape),
                        "dtype": inp.dtype,
                    }
                )
            else:
                self._input_info.append({})

        debug_mode = os.environ.get("DOCC_DEBUG")
        debug_dump = bool(debug_mode)

        # Build SDFG if not already done
        if self._sdfg is None:
            self._sdfg = self.to_sdfg(output_folder, debug_dump)

        sdfg = self._sdfg

        if debug_dump:
            sdfg.dump(output_folder, "mlir0.parsed", dump_dot=True)

        sdfg.validate()
        sdfg.expand()

        if debug_dump:
            sdfg.dump(output_folder, "mlir1.expanded", dump_dot=True)
        sdfg.simplify()
        if debug_dump:
            sdfg.dump(output_folder, "mlir2.opt", dump_dot=True)

        if self.target != "none":
            sdfg.normalize()

        if debug_dump or instrumentation_mode or capture_args:
            sdfg.dump(
                output_folder,
                "mlir3.norm",
                dump_dot=debug_dump,
                dump_json=True,
                record_for_instrumentation=True,
            )

        # Schedule if target is specified
        if self.target != "none":
            custom_schedule_fn = get_target_schedule_fn(self.target)
            if custom_schedule_fn is not None:
                custom_schedule_fn(
                    sdfg, self.category, {"remote_tuning": self.remote_tuning}
                )
            else:
                sdfg.schedule(self.target, self.category, self.remote_tuning)

        self.last_sdfg = sdfg

        if debug_dump:
            sdfg.dump(output_folder, "mlir4.post_sched", dump_dot=True)

        # Compile to shared library
        custom_compile_fn = get_target_compile_fn(self.target)
        if custom_compile_fn is not None:
            lib_path = custom_compile_fn(
                sdfg, output_folder, instrumentation_mode, capture_args, {}
            )
        else:
            lib_path = sdfg._compile(
                output_folder=output_folder,
                target=self.target,
                instrumentation_mode=instrumentation_mode,
                capture_args=capture_args,
            )

        # Build shape sources from input info
        shape_sources = []
        for i, info in enumerate(self._input_info):
            if "shape" in info:
                for dim_idx in range(len(info["shape"])):
                    shape_sources.append((i, dim_idx))

        # Create CompiledSDFG
        compiled = CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources=shape_sources,
        )

        # Cache
        if original_output_folder is None:
            self.cache[cache_key] = compiled

        self._compiled = compiled
        return compiled

    def to_sdfg(
        self,
        output_folder: Optional[str] = None,
        debug_dump: bool = False,
    ) -> StructuredSDFG:
        try:
            from torch_mlir import fx
        except ImportError:
            raise ImportError(
                "torch-mlir is required for importing torch models. "
                "Please install it with 'pip install torch-mlir'."
            )

        # Determine example input
        if self.example_input is None:
            raise ValueError("No example input provided for SDFG conversion.")

        # Import torch model to MLIR using torch-mlir FX
        # example_input may be a single tensor or a tuple of tensors;
        # fx.export_and_import expects them as positional *args.
        if isinstance(self.example_input, tuple):
            torch_mlir = fx.export_and_import(
                self.model, *self.example_input, output_type="linalg_on_tensors"
            )
        else:
            torch_mlir = fx.export_and_import(
                self.model, self.example_input, output_type="linalg_on_tensors"
            )
        torch_mlir = str(torch_mlir)

        # Translate to Structured SDFG
        mlir_module = MLIRModule(torch_mlir)
        mlir_module.convert()

        # Dump the MLIR code to a file for inspection
        if debug_dump and output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            with open(f"{output_folder}/{self.name}_imported.mlir", "w") as f:
                f.write(torch_mlir)
        sdfg_str = mlir_module.translate()
        try:
            sdfg = StructuredSDFG.parse(sdfg_str)
        except RuntimeError:
            if output_folder is not None:
                os.makedirs(output_folder, exist_ok=True)
                with open(
                    f"{output_folder}/{self.name}_parse_failed.sdfg.json", "w"
                ) as f:
                    f.write(sdfg_str)
            raise

        self._sdfg = sdfg
        return sdfg

    def _convert_inputs(self, args: tuple) -> tuple:
        import torch
        import numpy as np

        converted = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Ensure contiguous and convert to numpy
                arr = arg.detach().cpu().contiguous().numpy()
                converted.append(arr)
            elif isinstance(arg, np.ndarray):
                converted.append(arg)
            else:
                converted.append(arg)

        return tuple(converted)

    def _convert_outputs(
        self, result: Any, original_args: tuple, return_shape: tuple
    ) -> Any:
        import torch
        import numpy as np

        # Determine target device from input
        device = torch.device("cpu")
        for arg in original_args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break

        def convert_single(val, return_shape):
            if isinstance(val, np.ndarray):
                val = val.reshape(return_shape)
                return torch.from_numpy(val).to(device)
            elif isinstance(val, torch.Tensor):
                return val.reshape(return_shape).to(device)
            else:
                # Handle ctypes pointers (e.g. LP_c_float from CompiledSDFG)
                import ctypes

                if hasattr(val, "contents") and hasattr(val, "_type_"):
                    import math

                    num_elements = math.prod(return_shape)
                    ctype = val._type_
                    dtype_map = {
                        ctypes.c_float: np.float32,
                        ctypes.c_double: np.float64,
                        ctypes.c_int: np.int32,
                        ctypes.c_long: np.int64,
                    }
                    np_dtype = dtype_map.get(ctype, np.float32)
                    # Cast pointer to a ctypes array of the full size,
                    # then read with np.frombuffer to get all elements
                    ArrayType = ctype * num_elements
                    arr_ptr = ctypes.cast(val, ctypes.POINTER(ArrayType))
                    arr = np.frombuffer(arr_ptr.contents, dtype=np_dtype).copy()
                    arr = arr.reshape(return_shape)
                    return torch.from_numpy(arr).to(device)
                return val

        if isinstance(result, tuple):
            return tuple(convert_single(r, return_shape) for r in result)
        else:
            return convert_single(result, return_shape)

    def _get_cache_key(self, example_input: Any) -> str:
        import torch

        if not isinstance(example_input, tuple):
            inputs = (example_input,)
        else:
            inputs = example_input

        key_parts = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                key_parts.append(f"tensor({inp.shape},{inp.dtype})")
            else:
                key_parts.append(f"scalar({type(inp).__name__})")

        return "|".join(key_parts)


def compile_torch(
    model,
    example_input,
    target: str = "none",
    category: str = "server",
) -> CompiledSDFG:
    return TorchProgram(
        model,
        example_input=example_input,
        target=target,
        category=category,
    )


# ============================================================================
# torch.compile backend registration
# ============================================================================

# Global options for the docc backend (can be set before calling torch.compile)
_backend_options = {
    "target": "none",
    "category": "server",
}


def set_backend_options(target: str = "none", category: str = "server"):
    """Set global options for the docc torch.compile backend.

    Call this before using torch.compile(backend="docc") to configure
    the compilation target and category.

    Args:
        target: Compilation target ("none", "cuda", "openmp", etc.)
        category: Target category ("server", "server", etc.)

    Example:
        >>> from docc.torch import set_backend_options
        >>> set_backend_options(target="cuda", category="server")
        >>> compiled_model = torch.compile(model, backend="docc")
    """
    _backend_options["target"] = target
    _backend_options["category"] = category


def _docc_dynamo_compiler(gm, example_inputs):
    """Dynamic Compiler based on TorchProgram (inference only)."""
    import torch

    if len(example_inputs) == 1:
        example_input = example_inputs[0]
    else:
        example_input = tuple(example_inputs)

    program = TorchProgram(
        gm,
        example_input=example_input,
        target=_backend_options["target"],
        category=_backend_options["category"],
    )

    def compiled_fn(*args):
        result = program(*args)
        if isinstance(result, (tuple, list)):
            return result
        return (result,)

    return compiled_fn


def _docc_aot_compiler(gm, example_inputs):
    """AOTAutograd Compiler based on TorchProgram (inference and training)."""
    from functorch.compile import make_boxed_func

    import torch

    if len(example_inputs) == 1:
        example_input = example_inputs[0]
    else:
        example_input = tuple(example_inputs)

    program = TorchProgram(
        gm,
        example_input=example_input,
        target=_backend_options["target"],
        category=_backend_options["category"],
    )

    def compiled_fn(*args):
        result = program(*args)
        if isinstance(result, (tuple, list)):
            return result
        return (result,)

    return make_boxed_func(compiled_fn)


def _needs_autograd(gm, example_inputs):
    """Detect whether the current compilation requires autograd support.

    Uses torch.is_grad_enabled() as the signal. For inference, users should
    wrap calls in torch.no_grad() — this is standard PyTorch convention.
    Dynamo does not propagate model.eval() to the GraphModule, and lifted
    parameters always have requires_grad=True, so neither of those are
    reliable signals.
    """
    import torch

    return torch.is_grad_enabled()


def _docc_backend(gm, example_inputs):
    """Unified docc backend for torch.compile.

    Automatically selects the compilation strategy at runtime:
    - Inference (no grad): direct Dynamo path (faster compile, no overhead)
    - Training (grad required): AOTAutograd path (traces forward + backward)

    Usage:
        torch.compile(model, backend="docc")
    """
    if _needs_autograd(gm, example_inputs):
        from torch._dynamo.backends.common import aot_autograd

        aot_backend = aot_autograd(
            fw_compiler=_docc_aot_compiler,
            bw_compiler=_docc_aot_compiler,
        )
        return aot_backend(gm, example_inputs)
    else:
        return _docc_dynamo_compiler(gm, example_inputs)


def _register_backend():
    """Register the docc backend with torch.compile.

    This is called automatically when the module is imported, but only
    if torch._dynamo is available.
    """
    try:
        import torch._dynamo

        torch._dynamo.register_backend(name="docc")(_docc_backend)
    except ImportError:
        # torch._dynamo not available (older PyTorch version)
        pass
    except Exception:
        # Registration failed for some other reason, silently ignore
        pass


# Register the backend on module import
_register_backend()
