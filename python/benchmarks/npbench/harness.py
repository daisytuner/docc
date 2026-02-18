import argparse
import inspect
import sys
import time
import numpy as np
import pytest
import docc.python


def _get_func_param_names(func):
    """Get the parameter names of a function, excluding **kwargs."""
    sig = inspect.signature(func)
    return [
        name
        for name, param in sig.parameters.items()
        if param.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]


def _filter_params_for_func(func, all_params):
    """Filter params dict to only include parameters the function accepts."""
    param_names = _get_func_param_names(func)
    return {k: v for k, v in all_params.items() if k in param_names}


def _build_kernel_args(kernel_func, all_params):
    """Build the argument list for kernel_func in the correct order."""
    param_names = _get_func_param_names(kernel_func)
    return [all_params[name] for name in param_names]


def _combine_params_with_init_returns(params, initialize_func, init_returns):
    """
    Combine original parameters with initialize function return values.

    Uses the initialize function's return annotation or infers names from
    the kernel function parameters that aren't in the original params.
    """
    # Start with a copy of original params
    combined = dict(params)

    # Get return annotation if available
    sig = inspect.signature(initialize_func)
    if sig.return_annotation != inspect.Signature.empty:
        # If there's a return annotation, try to use it
        # This would require the function to have typed returns like -> Tuple[...]
        pass

    # For now, we need to infer the names from the source or use a naming convention
    # We'll get the names from the initialize function's source if possible
    try:
        source = inspect.getsource(initialize_func)
        # Look for a return statement and extract variable names
        import re

        # Match "return var1, var2, ..." pattern
        match = re.search(
            r"return\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*)\s*$",
            source,
            re.MULTILINE,
        )
        if match:
            return_names = [name.strip() for name in match.group(1).split(",")]
            if not isinstance(init_returns, tuple):
                init_returns = (init_returns,)
            if len(return_names) == len(init_returns):
                for name, value in zip(return_names, init_returns):
                    combined[name] = value
                return combined
    except (OSError, TypeError):
        pass

    # Fallback: if we can't extract names, just return init_returns as a tuple
    # and the caller should handle it differently
    return combined, init_returns


class SDFGVerification:
    def __init__(self, verification: dict):
        self._verification = verification

    def verify(self, stats: dict) -> None:
        print(stats)
        for key, val in self._verification.items():
            if key not in stats:
                assert val == 0, f"Key {key} not found in stats"
            else:
                assert (
                    stats[key] == val
                ), f"Key {key} has value {stats[key]} but expected {val}"
        print("All verifications passed.")


def run_benchmark(initialize_func, kernel_func, parameters, name, args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=str, default="paper")
        parser.add_argument("--docc", action="store_true")
        parser.add_argument("--numpy", action="store_true")
        parser.add_argument("--target", type=str, default="none")
        parser.add_argument("--n_runs", type=int, default=10)
        args = parser.parse_args()

    if args.size not in parameters:
        print(f"Unknown size: {args.size}. Available sizes: {list(parameters.keys())}")
        return

    params = parameters[args.size]

    # Filter params to only include what initialize_func needs
    init_params = _filter_params_for_func(initialize_func, params)
    init_returns = initialize_func(**init_params)

    # Combine original params with initialize return values
    combined = _combine_params_with_init_returns(params, initialize_func, init_returns)
    if isinstance(combined, tuple):
        # Fallback case - couldn't extract return names
        combined, init_returns = combined
        print("Warning: Could not extract return names from initialize function")

    # Build kernel arguments in the correct order
    kernel_args = _build_kernel_args(kernel_func, combined)

    if args.numpy:
        # Create a copy of inputs for numpy execution to avoid modification
        inputs_ref = [x.copy() if isinstance(x, np.ndarray) else x for x in kernel_args]
        for _ in range(args.n_runs):
            start = time.time()
            # Execute the original python function (undecorated)
            kernel_func(*inputs_ref)
            end = time.time()
            print(f"Numpy execution time: {end - start:.6f} seconds")

    if args.docc:
        # Create a copy of inputs for docc execution
        inputs_docc = [
            x.copy() if isinstance(x, np.ndarray) else x for x in kernel_args
        ]
        # Execute the decorated function
        kernel_with_target = docc.python.native(
            kernel_func,
            target=args.target,
        )

        times = []
        start = time.time()
        kernel_with_target(*inputs_docc)
        end = time.time()
        times.append(end - start)
        print(f"Docc execution time: {end - start:.6f} seconds")

        for _ in range(args.n_runs):
            start = time.time()
            kernel_with_target(*inputs_docc)
            end = time.time()
            times.append(end - start)
            print(f"Docc execution time (cached): {end - start:.6f} seconds")

        # print(f"Average Docc execution time over {N+1} runs: {np.mean(times):.6f} seconds")
        # print(f"Average Docc execution time (cached) over {N} runs: {np.mean(times[1:]):.6f} seconds")


# docc.sdfg_rpc_context = docc.DaisytunerTransfertuningRpcContext.from_docc_config()


def run_pytest(
    initialize_func,
    kernel_func,
    parameters,
    target="none",
    verifier: SDFGVerification = None,
):
    if sys.platform == "darwin":
        if target == "cuda":
            return

    # Use the smallest size for testing
    size = "S"
    if "S" not in parameters:
        size = list(parameters.keys())[0]

    params = parameters[size]

    # Filter params to only include what initialize_func needs
    init_params = _filter_params_for_func(initialize_func, params)
    init_returns = initialize_func(**init_params)

    # Combine original params with initialize return values
    combined = _combine_params_with_init_returns(params, initialize_func, init_returns)
    if isinstance(combined, tuple):
        # Fallback case - couldn't extract return names
        combined, init_returns = combined
        raise ValueError(
            "Could not extract return names from initialize function. "
            "Ensure the return statement uses simple variable names."
        )

    # Build kernel arguments in the correct order
    kernel_args = _build_kernel_args(kernel_func, combined)

    # Run Python version
    inputs_ref = [x.copy() if isinstance(x, np.ndarray) else x for x in kernel_args]
    res_ref = kernel_func(*inputs_ref)

    # Run Docc version
    inputs_docc = [x.copy() if isinstance(x, np.ndarray) else x for x in kernel_args]

    kernel_with_target = docc.python.native(
        kernel_func,
        target=target,
        category="server",
    )
    res_docc = kernel_with_target(*inputs_docc)

    sdfg = kernel_with_target.last_sdfg
    stats = sdfg.loop_report()
    print(stats)  # {'FOR': 5, 'MAP': 2, 'CPU': 2, ...}
    assert stats is not None, "No stats found in SDFG."
    verifier.verify(stats)

    # Validate return values if they exist
    if res_ref is not None:
        if isinstance(res_ref, tuple):
            for i in range(len(res_ref)):
                np.testing.assert_allclose(
                    res_docc[i], res_ref[i], rtol=1e-5, atol=1e-8
                )
        else:
            np.testing.assert_allclose(res_docc, res_ref, rtol=1e-5, atol=1e-8)

    # Validate arguments (in-place modifications)
    for i in range(len(kernel_args)):
        if isinstance(kernel_args[i], np.ndarray):
            np.testing.assert_allclose(
                inputs_docc[i], inputs_ref[i], rtol=1e-5, atol=1e-8
            )
