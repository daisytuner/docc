from docc.benchmarks.rtl import try_ensure_instrumentation_ready

# Ready the counter backends (e.g. ROCm) early: they must initialize before the
# workload sets up its own runtime. Best-effort so importing docc never fails when
# instrumentation is not configured.
try_ensure_instrumentation_ready()


def check_pytorch_version(
    min_version: tuple[int, int, int], max_version: tuple[int, int, int]
) -> None:
    import torch

    str_version: list[str] = torch.__version__.split("+", 1)[0].split(".")
    if not all([elem.isnumeric() for elem in str_version]):
        import warnings

        warnings.warn(
            "Could not read PyTorch version number because at least one element is not numeric: "
            + ".".join(str_version)
        )

    int_version = tuple([int(elem) for elem in str_version])
    if int_version < min_version or int_version > max_version:
        import warnings

        warnings.warn(
            "Unsupported PyTorch version detected: "
            + ".".join(str_version)
            + "\nUsing this version might work, but cannot be guaranteed.\nSupported versions are between "
            + ".".join([str(elem) for elem in min_version])
            + " and "
            + ".".join([str(elem) for elem in max_version])
        )


check_pytorch_version((2, 8, 0), (2, 13, 0))


from docc.pytorch.pytorch_program import PyTorchProgram
from docc.pytorch.graph_parser import GraphParser
