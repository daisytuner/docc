import os
import getpass
import hashlib
import shutil
from typing import Any, Optional

from docc.sdfg import StructuredSDFG
from docc.mlir import MLIRModule
from docc.compiler import DoccProgram, CompiledSDFG

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
            hash_input = (
                f"{self.name}|{self.target}|{self.category}|{cache_key}".encode("utf-8")
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

        # Build SDFG if not already done
        if self._sdfg is None:
            self._sdfg = self.to_sdfg()

        sdfg = self._sdfg
        sdfg.validate()
        sdfg.expand()
        sdfg.simplify()

        if self.target != "none":
            sdfg.normalize()

        sdfg.dump(output_folder)

        # Schedule if target is specified
        if self.target != "none":
            sdfg.schedule(self.target, self.category, self.remote_tuning)

        self.last_sdfg = sdfg

        # Compile to shared library
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

    def to_sdfg(self) -> StructuredSDFG:
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
        torch_mlir = fx.export_and_import(
            self.model, self.example_input, output_type="linalg_on_tensors"
        )
        torch_mlir = str(torch_mlir)

        # Convert to SDFG dialect
        mlir_module = MLIRModule(torch_mlir)
        mlir_module.convert()

        sdfg_str = mlir_module.translate()
        sdfg = StructuredSDFG.parse(sdfg_str)

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
                return torch.from_numpy(val).to(device)
            return val

        if isinstance(result, tuple):
            return tuple(convert_single(r) for r in result)
        else:
            return convert_single(result)

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


def _docc_backend(gm: "torch.fx.GraphModule", example_inputs):
    """Backend function for torch.compile integration.

    This function is called by torch.compile when backend="docc" is specified.
    It compiles the FX graph to native code using the docc compiler.

    Args:
        gm: The torch.fx.GraphModule captured by dynamo
        example_inputs: List of example input tensors

    Returns:
        A callable that executes the compiled code
    """
    import torch

    # Convert example_inputs list to tuple for TorchProgram
    if len(example_inputs) == 1:
        example_input = example_inputs[0]
    else:
        example_input = tuple(example_inputs)

    # Create TorchProgram from the FX GraphModule
    program = TorchProgram(
        gm,
        example_input=example_input,
        target=_backend_options["target"],
        category=_backend_options["category"],
    )

    # Return the compiled callable
    return program


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
