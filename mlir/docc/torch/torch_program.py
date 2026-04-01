import os
import getpass
import hashlib
import shutil
from typing import Any, Optional, List
import time

from docc.compiler import CompiledSDFG, DoccProgram
from docc.compiler.target_registry import reset_target_registry
from docc.sdfg import StructuredSDFG
from docc.mlir import MLIRModule


def _filter_none_outputs(model) -> List[int]:
    """Filter None values from FX graph outputs.

    For AOTAutograd backward graphs, None indicates "no gradient for this input".
    torch-mlir cannot lower torch.constant.none to linalg, so we filter them out
    before export and restore them after execution.

    Args:
        model: A torch.fx.GraphModule to potentially modify

    Returns:
        List of positions where None was filtered out (empty if no changes)
    """
    import torch.fx

    if not isinstance(model, torch.fx.GraphModule):
        return []

    graph = model.graph

    # Find the output node
    output_node = None
    for node in graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return []

    # Get the return args (tuple)
    ret_args = output_node.args[0]
    if not isinstance(ret_args, tuple):
        return []

    # Check if any are None
    none_positions = []
    filtered_args = []
    for i, arg in enumerate(ret_args):
        if arg is None:
            none_positions.append(i)
        else:
            filtered_args.append(arg)

    if not none_positions:
        return []  # No Nones to filter

    # Rewrite the output node
    with graph.inserting_before(output_node):
        graph.output(tuple(filtered_args))
    graph.erase_node(output_node)

    # Recompile the graph module
    model.recompile()

    return none_positions


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
        force_rebuild: bool = False,
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
        self._frozen_buffer_args: list = []  # buffer tensors not frozen by torch-mlir
        self._none_output_positions: List[int] = (
            []
        )  # positions filtered from backward graph
        self.force_rebuild = force_rebuild

    def __call__(self, *args: Any) -> Any:
        import torch

        # Detect input type (torch or numpy)
        is_torch_input = any(isinstance(arg, torch.Tensor) for arg in args)

        # Compile if necessary
        if self._compiled is None:
            self._compiled = self.compile()

        # Prepend frozen buffer values (e.g. BatchNorm running_mean/var) that
        # torch-mlir left as function arguments instead of freezing as constants.
        if self._frozen_buffer_args:
            all_args = tuple(self._frozen_buffer_args) + args
        else:
            all_args = args

        # Convert inputs to numpy
        numpy_args = self._convert_inputs(all_args)

        # Execute - CompiledSDFG returns tuple for multi-output or single value
        result = self._compiled(*numpy_args)

        # Convert outputs back to torch if inputs were torch
        if is_torch_input:
            result = self._convert_outputs(result, args)

        return result

    def compile(
        self,
        output_folder: Optional[str] = None,
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
    ) -> CompiledSDFG:
        original_output_folder = output_folder

        compile_profile = os.environ.get("DOCC_PROFILE_COMPILE", "")
        if compile_profile:
            print("Compiling Torch Model>")
            compile_start_time = time.perf_counter()

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

        cached_available = cache_key in self.cache

        if original_output_folder and cached_available:
            if not self.force_rebuild:
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

        # Reuse already built binaries
        docc_reuse_binaries = os.environ.get("DOCC_REUSE_BINARIES")

        if not os.path.exists(output_folder) and docc_reuse_binaries:
            docc_reuse_binaries = None
        elif os.path.exists(output_folder) and not docc_reuse_binaries:
            shutil.rmtree(output_folder)

        # Populate input info from example input
        import torch

        self._input_info = []
        example_inputs = (
            tuple(self.example_input)
            if isinstance(self.example_input, (tuple, list))
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

        if docc_reuse_binaries:
            lib_path = f"{output_folder}/lib__docc_{self.name}.so"
            if not os.path.exists(lib_path):
                raise ValueError(
                    f"Tried reusing binary '{lib_path}' but does not exist"
                )
            sdfg_path = f"{output_folder}/__docc_{self.name}.py3.norm.json"
            if not os.path.exists(sdfg_path):
                raise ValueError(f"Tried loading SDFG '{sdfg_path}' but does not exist")
            sdfg = StructuredSDFG.from_file(sdfg_path)
            self._sdfg = sdfg
        else:
            # Build SDFG if not already done
            if self._sdfg is None:
                self._sdfg = self.to_sdfg(output_folder)

            sdfg = self._sdfg

            lib_path = self.sdfg_pipe(
                sdfg, output_folder, instrumentation_mode, capture_args
            )

        # Prepend buffer info for any buffers that torch-mlir left as
        # function arguments (detected in to_sdfg).
        if self._frozen_buffer_args:
            buffer_info = [
                {"shape": tuple(buf.shape), "dtype": buf.dtype}
                for buf in self._frozen_buffer_args
            ]
            self._input_info = buffer_info + self._input_info

        # Build shape sources from input info
        shape_sources = []
        for i, info in enumerate(self._input_info):
            if "shape" in info:
                for dim_idx in range(len(info["shape"])):
                    shape_sources.append((i, dim_idx))

        # Extract output_args metadata for multi-output support
        output_args_str = sdfg.metadata("output_args")
        output_args = output_args_str.split(",") if output_args_str else []

        # Extract output shapes from metadata (e.g., "_docc_ret_0_shape")
        output_shapes = {}
        for arg_name in output_args:
            shape_str = sdfg.metadata(f"{arg_name}_shape")
            if shape_str:
                # Parse shape string like "[2, 4]" to list of dimension strings
                dims = [
                    d.strip() for d in shape_str.strip("[]").split(",") if d.strip()
                ]
                output_shapes[arg_name] = dims

        # Create CompiledSDFG with output_args for multi-output support
        compiled = CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources=shape_sources,
            output_args=output_args,
            output_shapes=output_shapes,
        )

        # Cache
        if original_output_folder is None:
            self.cache[cache_key] = compiled

        self._compiled = compiled

        if compile_profile:
            docc_compile_time = time.perf_counter() - compile_start_time
            print(f">DOCC compile done: {docc_compile_time:.4f} s")
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

        # Import torch model to MLIR using torch-mlir FX.
        # torch-mlir's import_frozen_program freezes parameters as MLIR constants
        # but may leave buffers (e.g. BatchNorm running_mean/var) as function
        # arguments with newer PyTorch versions. Detect such buffers so __call__
        # can prepend their values alongside user inputs.
        import torch

        example_inputs = (
            tuple(self.example_input)
            if isinstance(self.example_input, (tuple, list))
            else (self.example_input,)
        )

        self._frozen_buffer_args = []
        try:
            prog = torch.export.export(self.model, example_inputs)
            sig = prog.graph_signature
            if hasattr(prog, "constants"):
                for spec in sig.input_specs:
                    if spec.kind == torch.export.graph_signature.InputKind.BUFFER:
                        obj = self.model
                        for part in spec.target.split("."):
                            obj = getattr(obj, part)
                        self._frozen_buffer_args.append(obj.detach().cpu().contiguous())
        except Exception:
            self._frozen_buffer_args = []

        # Filter None outputs from FX graph (AOTAutograd backward graphs use None
        # to indicate "no gradient for this input"). torch-mlir cannot lower
        # torch.constant.none, so we filter them here and restore after execution.
        self._none_output_positions = _filter_none_outputs(self.model)

        torch_mlir = fx.export_and_import(
            self.model,
            *example_inputs,
            output_type="linalg_on_tensors",
            func_name=self.name,
        )
        torch_mlir = str(torch_mlir)

        # Dump the MLIR code to a file for inspection
        if self.debug_dump and output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            with open(f"{output_folder}/{self.name}_imported.mlir", "w") as f:
                f.write(torch_mlir)

        # Translate to Structured SDFG
        mlir_module = MLIRModule(torch_mlir)
        mlir_module.convert()

        # Dump the MLIR code to a file for inspection after conversion
        if self.debug_dump and output_folder is not None:
            torch_mlir_converted = mlir_module.to_string()
            with open(f"{output_folder}/{self.name}_converted.mlir", "w") as f:
                f.write(torch_mlir_converted)

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

    def _convert_outputs(self, result: Any, original_args: tuple) -> Any:
        import torch
        import numpy as np

        # Determine target device from input
        device = torch.device("cpu")
        for arg in original_args:
            if isinstance(arg, torch.Tensor):
                device = arg.device
                break

        def convert_single(val):
            if isinstance(val, np.ndarray):
                return torch.from_numpy(val.copy()).to(device)
            elif isinstance(val, torch.Tensor):
                return val.to(device)
            elif isinstance(val, (int, float)):
                return torch.tensor(val, device=device)
            else:
                return val

        if isinstance(result, tuple):
            converted = list(convert_single(r) for r in result)
        else:
            converted = [convert_single(result)]

        # Restore None values at recorded positions (filtered from backward graph)
        if self._none_output_positions:
            for pos in sorted(self._none_output_positions):
                converted.insert(pos, None)

        # Return single value if only one output, otherwise tuple
        if len(converted) == 1:
            return converted[0]
        return tuple(converted)

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
    force_rebuild: bool = False,
) -> CompiledSDFG:
    return TorchProgram(
        model,
        example_input=example_input,
        target=target,
        category=category,
        force_rebuild=force_rebuild,
    )


# ============================================================================
# torch.compile backend registration
# ============================================================================

# Global options for the docc backend (can be set before calling torch.compile)
_backend_options = {
    "target": "none",
    "category": "server",
}


def reset_backend_options():
    _backend_options["target"] = "none"
    _backend_options["category"] = "server"
    reset_target_registry()


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
