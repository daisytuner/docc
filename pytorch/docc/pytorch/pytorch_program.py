import torch
import torch.fx
import torch._dynamo
import torch.export
import torch.utils._pytree
import numpy as np
import ml_dtypes
import os
import time
import hashlib
import getpass
import shutil
from typing import Any

from docc.compiler import DoccProgram, CompiledSDFG
from docc.sdfg import StructuredSDFG, DoccMetrics

from docc.pytorch.graph_parser import GraphParser


class PyTorchProgram(DoccProgram):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_input: Any = None,
        target: str = "none",
        category: str = "server",
        remote_tuning: bool = False,
        instrumentation_mode: str | None = None,
        capture_args: bool | None = None,
        force_rebuild: bool = False,
        name: str | None = None,
    ) -> None:
        # Determine name from model
        if name is None:
            if hasattr(gm, "__class__"):
                sdfg_name: str = gm.__class__.__name__
            else:
                sdfg_name: str = "torch_model"
        else:
            sdfg_name: str = name

        # Sanitize name: remove characters that are invalid in filesystem paths
        import re as _re

        sdfg_name: str = _re.sub(r"[<>:\"/\\|?*]", "_", sdfg_name)

        super().__init__(
            name=sdfg_name,
            target=target,
            category=category,
            instrumentation_mode=instrumentation_mode,
            capture_args=capture_args,
            remote_tuning=remote_tuning,
        )

        self.gm: torch.fx.GraphModule = gm
        if example_input is None:
            self.example_input: tuple[Any, ...] | None = None
        elif isinstance(example_input, tuple):
            self.example_input: tuple[Any, ...] | None = example_input
        else:
            self.example_input: tuple[Any, ...] | None = (example_input,)
        self._sdfg: StructuredSDFG | None = None
        self._compiled: CompiledSDFG | None = None
        self._input_info: list[dict[str, tuple[int, ...] | torch.dtype]] = (
            []
        )  # [(shape, dtype), ...] for each input
        self._output_info: list[tuple[torch.Size, torch.dtype]] = (
            []
        )  # [(shape, dtype), ...] for each output
        # PyTree spec describing the original PyTorch output structure. Loaded
        # from SDFG metadata after compilation and used to reassemble the flat
        # outputs into the exact structure eager PyTorch would return.
        self._out_spec: torch.utils._pytree.TreeSpec | None = None
        # Permutation mapping the compiled artifact's output order (SDFG function
        # signature order) to the PyTorch graph output order (== pytree leaf
        # order). Computed at compile time; None when no reordering is needed.
        self._output_perm: list[int] | None = None
        self.force_rebuild: bool = force_rebuild

    def __call__(self, *args: Any) -> Any:
        # Detect input type (torch or numpy)
        is_torch_input: bool = any(isinstance(arg, torch.Tensor) for arg in args)

        # Compile if necessary
        if self._compiled is None:
            compiled_sdfg: CompiledSDFG = self.compile()
            self._compiled = compiled_sdfg
        else:
            compiled_sdfg: CompiledSDFG = self._compiled

        # Device-resident artifacts consume/produce device pointers directly:
        # pass tensors straight through. CompiledSDFG runs CUDA tensors zero-copy
        # and copies CPU tensors to the device (with a one-time warning).
        if compiled_sdfg.device_resident:
            if self._compiled is None:
                raise ValueError("Compiled SDFG is None, cannot execute.")
            result = self._compiled(*args)
            if is_torch_input:
                result: Any = self._convert_outputs(result, args)
            return result

        # Non-device-resident artifact: CUDA tensors cannot run on the host.
        if any(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
            raise TypeError(
                "CUDA torch tensors were provided, but this artifact is not "
                "device-resident. GPU tensors can only be used with a "
                "device-resident artifact (a fully-offloadable model compiled "
                "for a GPU target). Move the tensors to the host (`.cpu()`) to "
                "run on this artifact."
            )

        # Host execution: convert CPU tensors to numpy, run, convert back.
        numpy_args: Any = self._convert_inputs(args)
        if self._compiled is None:
            raise ValueError("Compiled SDFG is None, cannot execute.")
        result = self._compiled(*numpy_args)
        if is_torch_input:
            result = self._convert_outputs(result, args)
        return result

    def compile(
        self,
        output_folder: str | None = None,
        instrumentation_mode: str | None = None,
        capture_args: bool | None = None,
        remote_tuning: bool | None = None,
    ) -> CompiledSDFG:
        original_output_folder: str | None = output_folder

        metrics = DoccMetrics()
        compile_start_time = time.perf_counter()
        metrics.add_frontend_source_info("torch")
        metrics.add_metric("model_name", self.name, "source")

        # Resolve options
        instrumentation_mode, capture_args, remote_tuning = (
            self._resolve_compile_options(
                instrumentation_mode, capture_args, remote_tuning
            )
        )

        # Determine example input
        if self.example_input is None:
            raise ValueError(
                "No example input provided. Either provide example_input during "
                "initialization or pass example inputs to compile()."
            )

        # Generate cache key
        cache_key: str = self._get_cache_key(self.example_input)

        # In-memory cache key: the structural cache key plus the resolved compile
        # options, so repeated in-process compiles with different
        # instrumentation/arg-capture/remote-tuning do not alias to the first
        # built binary (the on-disk hash already accounts for these options).
        mem_cache_key: str = (
            f"{cache_key}|{capture_args}|{instrumentation_mode}|{remote_tuning}"
        )

        cached_available: bool = mem_cache_key in self.cache

        if original_output_folder and cached_available:
            if not self.force_rebuild:
                return self.cache[mem_cache_key]

        # Determine output folder
        if output_folder is None:
            # Include model-specific code in hash to distinguish different models
            # that share the same class name and input shapes.
            # - FX GraphModules (from torch.compile): use the FX graph code
            # - Regular nn.Modules (from compile_torch): use forward() source
            model_code: str = ""
            try:
                model_code: str = self.gm.graph.python_code("self").src
            except Exception:
                pass
            if not model_code:
                try:
                    import inspect

                    model_code: str = inspect.getsource(type(self.gm).forward)
                except Exception:
                    pass

            hash_input: bytes = (
                f"{self.name}|{self.target}|{self.category}|{cache_key}|{model_code}|{capture_args}|{instrumentation_mode}|{remote_tuning}".encode(
                    "utf-8"
                )
            )
            stable_id: str = hashlib.sha256(hash_input).hexdigest()[:16]

            docc_tmp: str | None = os.environ.get("DOCC_TMP")
            if docc_tmp:
                output_folder_path: str = f"{docc_tmp}/{self.name}-{stable_id}"
            else:
                user: str = os.getenv("USER", "")
                if not user:
                    user: str = getpass.getuser()
                output_folder_path: str = f"/tmp/{user}/DOCC/{self.name}-{stable_id}"
        else:
            output_folder_path: str = output_folder

        # Reuse already built binaries
        docc_reuse_binaries: str | None = os.environ.get("DOCC_REUSE_BINARIES")

        # Reuse already generated sources (recompile without regenerating them).
        # Unlike binary reuse this still runs the full pipeline, but the build
        # step recompiles the existing source files instead of overwriting them.
        docc_reuse_sources: str | None = os.environ.get("DOCC_REUSE_SOURCES")

        if (docc_reuse_binaries or docc_reuse_sources) and not self.debug_dump:
            self.debug_dump: bool = True  # Required for source reuse

        if not os.path.exists(output_folder_path) and docc_reuse_sources:
            docc_reuse_sources: str | None = None

        if not os.path.exists(output_folder_path) and docc_reuse_binaries:
            docc_reuse_binaries: str | None = None
        elif (
            os.path.exists(output_folder_path)
            and not docc_reuse_binaries
            and not docc_reuse_sources
        ):
            shutil.rmtree(output_folder_path)

        # Populate input info from example input
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
            lib_path = f"{output_folder_path}/lib__docc_{self.name}.so"
            if not os.path.exists(lib_path):
                raise ValueError(
                    f"Tried reusing binary '{lib_path}' but does not exist"
                )
            sdfg_path = f"{output_folder_path}/__docc_{self.name}.py5.post_sched.json"
            if not os.path.exists(sdfg_path):
                raise ValueError(f"Tried loading SDFG '{sdfg_path}' but does not exist")
            sdfg = StructuredSDFG.from_file(sdfg_path)
            self._sdfg = sdfg

            # The cached .so embeds a device-resident (or host) calling
            # convention chosen at compile time and recorded in the SDFG
            # metadata. Restore that decision so we marshal arguments the same
            # way; otherwise a device-resident binary would be fed host pointers
            # via the host path -> double free.
            self._device_resident = sdfg.metadata("device_resident") == "1"
            backend = sdfg.metadata("device_backend")
            self._device_backend = backend or None
        elif docc_reuse_sources:

            sdfg_path = f"{output_folder_path}/__docc_{self.name}.py5.post_sched.json"
            if not os.path.exists(sdfg_path):
                raise ValueError(f"Tried loading SDFG '{sdfg_path}' but does not exist")
            sdfg = StructuredSDFG.from_file(sdfg_path)

            main_file = f"{output_folder_path}/__docc_{self.name}.cpp"
            if not os.path.exists(main_file):
                raise ValueError(
                    f"Tried reusing sources '{main_file}' but does not exist"
                )

            lib_path = self.sdfg_pipe(
                sdfg,
                output_folder_path,
                instrumentation_mode,
                capture_args,
                remote_tuning,
                reuse_sources=True,
            )
        else:
            # Build SDFG if not already done
            if self._sdfg is None:
                self._sdfg = self.to_sdfg(output_folder_path)

            parse_sdfg_time = time.perf_counter() - compile_start_time
            metrics.add_metric(
                "parse_to_sdfg_time_ms", round(parse_sdfg_time * 1000), "compile_times"
            )

            sdfg = self._sdfg

            lib_path = self.sdfg_pipe(
                sdfg,
                output_folder_path,
                instrumentation_mode,
                capture_args,
                remote_tuning,
                metrics=metrics,
            )

        # Build shape sources from input info
        shape_sources = []
        for i, info in enumerate(self._input_info):
            if "shape" in info:
                shape: tuple[int, ...] | torch.dtype = info["shape"]
                if not isinstance(shape, tuple):
                    raise TypeError(
                        "Expected tuple type for shape information but got: "
                        + str(type(shape))
                    )
                for dim_idx in range(len(shape)):
                    shape_sources.append((i, dim_idx))

        # Extract output_args metadata for multi-output support
        if sdfg is None:
            raise ValueError("SDFG is None, cannot extract output_args metadata")
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

        # Load the PyTorch output pytree spec (if present) so results can be
        # reassembled into the exact structure eager PyTorch would return.
        # Reading from SDFG metadata (rather than the live ExportedProgram) means
        # reused binaries reconstruct outputs correctly as well.
        out_spec_str = sdfg.metadata("output_pytree_spec")
        if out_spec_str:
            self._out_spec = torch.utils._pytree.treespec_loads(out_spec_str)
        else:
            self._out_spec = None

        # CompiledSDFG (with sort_output_args=False) returns outputs in SDFG
        # function-signature order, i.e. the order the output containers appear
        # in sdfg.arguments. The PyTorch graph output order (and thus the pytree
        # leaf order) is the order given by output_args. Precompute a permutation
        # that reorders the artifact's flat outputs back into output_args order
        # so results line up with the pytree spec and match eager PyTorch.
        self._output_perm = None
        if len(output_args) > 1 and hasattr(sdfg, "arguments"):
            output_args_set = set(output_args)
            signature_order = [n for n in sdfg.arguments if n in output_args_set]
            # Only reorder when the sets match 1:1 (no duplicate outputs) and the
            # order actually differs; otherwise the flat order is already correct.
            if (
                len(signature_order) == len(output_args)
                and set(signature_order) == output_args_set
                and signature_order != output_args
            ):
                self._output_perm = [
                    signature_order.index(name) for name in output_args
                ]

        # Create CompiledSDFG with output_args for multi-output support

        compiled = CompiledSDFG(
            lib_path,
            sdfg,
            shape_sources=shape_sources,
            output_args=output_args,
            output_shapes=output_shapes,
            device_resident=self._device_resident,
            device_backend=self._device_backend,
            target=self.target,
            sort_output_args=False,
        )

        # Cache
        if original_output_folder is None:
            self.cache[mem_cache_key] = compiled

        self._compiled = compiled

        if not docc_reuse_binaries:
            # Record compile time in milliseconds, rounded to the nearest integer.
            compile_time_ms = round((time.perf_counter() - compile_start_time) * 1000)
            metrics.add_metric("compile_time_ms", compile_time_ms, "compile_times")

        metrics.capture_env_vars()
        metrics.append_to(output_folder_path)

        return compiled

    def to_sdfg(self, output_folder: str | None = None) -> StructuredSDFG:
        # Determine example input
        if self.example_input is None:
            raise ValueError("No example input provided for SDFG conversion.")

        # Use torch.export and run decompositions
        exported_program: torch.export.ExportedProgram = torch.export.export(
            self.gm, self.example_input
        )
        ir: torch.export.ExportedProgram = exported_program.run_decompositions(
            decomp_table=None
        )

        # Dump the IR to a file for inspection
        if self.debug_dump and output_folder is not None:
            os.makedirs(output_folder, exist_ok=True)
            with open(f"{output_folder}/{self.name}.py", "w") as f:
                f.write("import torch\nimport torch.nn\nimport torch.ops\n\n")
                f.write(ir.graph_module.print_readable(print_output=False))

        # Run graph parser
        parser: GraphParser = GraphParser(self.name, ir, self.example_input)
        parser.parse()
        sdfg = parser.to_sdfg()

        try:
            sdfg.validate()
        except RuntimeError:
            if output_folder is not None:
                os.makedirs(output_folder, exist_ok=True)
                with open(
                    f"{output_folder}/{self.name}_parse_failed.sdfg.json", "w"
                ) as f:
                    f.write(sdfg.to_json())
            raise

        self._sdfg: StructuredSDFG | None = sdfg
        return sdfg

    def _convert_inputs(self, args: tuple) -> tuple[np.ndarray, ...]:
        converted: list[np.ndarray] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Ensure contiguous and convert to numpy
                contiguous_arg: torch.Tensor = arg.detach().cpu().contiguous()
                if arg.dtype == torch.bfloat16:
                    arr = (
                        contiguous_arg.view(dtype=torch.uint16)
                        .numpy()
                        .view(dtype=ml_dtypes.bfloat16)
                    )
                else:
                    arr: np.ndarray = contiguous_arg.numpy()
                converted.append(arr)
            elif isinstance(arg, np.ndarray):
                converted.append(arr)
            else:
                raise ValueError("Only allow torch.Tensor and np.ndarray for now")
        return tuple(converted)

    def _convert_outputs(self, result: Any, original_args: tuple) -> Any:
        # Determine target device from input
        device: torch.device = torch.device("cpu")
        for arg in original_args:
            if isinstance(arg, torch.Tensor):
                device: torch.device = arg.device
                break

        # Fast path: check if we're on CPU (avoid device transfer overhead)
        is_cpu: bool = device.type == "cpu"

        def convert_single(val: Any) -> torch.Tensor:
            if isinstance(val, np.ndarray):
                # torch.from_numpy shares memory - this is fast
                # For non-CPU devices, .to(device) creates a copy anyway
                # For CPU, the shared memory is fine since CompiledSDFG
                # allocates new buffers on each call
                if val.dtype == ml_dtypes.bfloat16:
                    t = torch.from_numpy(val.view(dtype=np.uint16)).view(
                        dtype=torch.bfloat16
                    )
                else:
                    t = torch.from_numpy(val)
                if not is_cpu:
                    t = t.to(device)
                return t
            elif isinstance(val, torch.Tensor):
                return val if is_cpu else val.to(device)
            elif isinstance(val, (int, float)):
                return torch.tensor(val, device=device)
            elif hasattr(val, "__cuda_array_interface__"):
                # Device-resident output (e.g. cupy array): zero-copy to torch.
                t = torch.from_dlpack(val)
                if t.device != device:
                    t = t.to(device)
                return t
            else:
                return val

        if isinstance(result, tuple):
            converted: list[torch.Tensor] = [convert_single(r) for r in result]
        else:
            converted: list[torch.Tensor] = [convert_single(result)]

        # Reorder the artifact's flat outputs (SDFG signature order) into the
        # PyTorch graph output order (== pytree leaf order) before reassembly.
        if self._output_perm is not None and len(converted) == len(self._output_perm):
            converted = [converted[p] for p in self._output_perm]

        # Reassemble the flat outputs into the original PyTorch structure (nested
        # tuples, dicts, dataclasses, single tensor, None leaves, ...) using the
        # pytree spec captured at compile time. The compiled artifact preserves
        # the flat leaf order (sort_output_args=False), so it lines up with the
        # spec's leaves. Fall back to the flat single/tuple behavior when no spec
        # is available (e.g. legacy artifacts) or the leaf count does not match.
        if self._out_spec is not None and len(converted) == self._out_spec.num_leaves:
            return torch.utils._pytree.tree_unflatten(converted, self._out_spec)

        # Return single value if only one output, otherwise tuple
        if len(converted) == 1:
            return converted[0]
        return tuple(converted)

    def _get_cache_key(self, example_input: Any) -> str:
        import torch

        if not isinstance(example_input, tuple):
            inputs: tuple = (example_input,)
        else:
            inputs: tuple = example_input

        key_parts: list[str] = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                key_parts.append(f"tensor({inp.shape},{inp.dtype})")
            else:
                key_parts.append(f"scalar({type(inp).__name__})")

        return "|".join(key_parts)


def _docc_get_backend_options(
    options: None | dict[str, str | bool],
) -> tuple[str, str, bool]:
    target: str = "none"
    category: str = "server"
    remote_tuning: bool = False
    if options:
        if "target" in options and type(options["target"]) == str:
            target: str = options["target"]
        if "category" in options and type(options["category"]) == str:
            category: str = options["category"]
        if "remote_tuning" in options and type(options["remote_tuning"]) == bool:
            remote_tuning: bool = options["remote_tuning"]
    return target, category, remote_tuning


def _docc_inference_compiler(
    gm: torch.fx.GraphModule,
    example_inputs,
    backend_options: None | dict[str, str | bool],
):
    """Dynamic Compiler based on TorchProgram (inference only)."""

    if len(example_inputs) == 1:
        example_input = example_inputs[0]
    else:
        example_input = tuple(example_inputs)

    target, category, remote_tuning = _docc_get_backend_options(backend_options)
    program = PyTorchProgram(
        gm,
        example_input=example_input,
        target=target,
        category=category,
        remote_tuning=remote_tuning,
    )

    def compiled_fn(*args):
        result = program(*args)
        if isinstance(result, (tuple, list)):
            return result
        return (result,)

    return compiled_fn


def _docc_backend(gm, example_inputs, *, options=None):
    """Unified docc backend for torch.compile.

    Automatically selects the compilation strategy at runtime:
    - Inference (no grad): direct Dynamo path (faster compile, no overhead)
    - Training (grad required): AOTAutograd path (traces forward + backward)

    Usage:
        torch.compile(model, backend="docc")
    """
    if torch.is_grad_enabled():
        # from torch._dynamo.backends.common import aot_autograd

        # target, category, remote_tuning = _docc_get_backend_options(options)
        # aot_backend = aot_autograd(
        #     fw_compiler=_docc_aot_compiler_wrapper(target, category, remote_tuning),
        #     bw_compiler=_docc_aot_compiler_wrapper(target, category, remote_tuning),
        # )
        # return aot_backend(gm, example_inputs)
        raise RuntimeError("Currently AOTautograd is not supported")
    else:
        return _docc_inference_compiler(gm, example_inputs, options)


# Register the backend on module import
torch._dynamo.register_backend(name="docc")(_docc_backend)
