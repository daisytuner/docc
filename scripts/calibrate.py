"""Measures the processor model of an Axelera AI processor, Metis or Europa, on a card of that generation, as
sdfglib-auto's benchmarks/cuda and benchmarks/rocm do for GPUs: hand-written AxScript kernels, compiled by alto and
timed with the AI cores' cycle counters (the profiler's DEVICE_MAIN level), in the processor model's units (FLOP/us,
bytes/us, us). It needs only a Voyager SDK checkout and its virtual environment, not docc.

    python calibrate.py --arch europa                       # all kernels; writes axelera-europa.json
    python calibrate.py --arch metis mvm ddr                # some of them
    python calibrate.py --arch europa --compile-only        # compiles every kernel, without a card
    VOYAGER_SDK=/path/to/voyager-sdk python calibrate.py ... # if Python's environment is not the SDK's

Kernels, each on one AI core unless stated:

- ``dpu``: float multiplications on the DPU, 32 per PWORD, as one dependent chain and as eight independent chains, less
  the same datapaths without them: cycles per DPU instruction, and the float peak on all cores.
- ``mvm``: int8 matmuls of 512 rows against weights resident in the IMC, 1024 x 1024, 2048 x 512 and 4096 x 256,
  repeated: cycles per row, and the int8 peak on all cores (a multiply-accumulate is 2 FLOP).
- ``l1``: a datapath's two IFDs and ODR streaming 128 x 1024 float32 through L1, repeated.
- ``l2``: L2 -> L1 copies of 2 MiB per core, after one DDR -> L2 copy, repeated, on 1, 2, 4, ... cores.
- ``ddr``: DDR -> L1 copies of 2 MiB per core, contiguous, on 1, 2, 4, ... cores.
- ``launch``: a module that copies one PWORD into L1 and back: what a module costs beyond its work.

The profiler can collect no samples from a run; such measurements print as missing, and the model uses those on fewer
cores or leaves the field 0. The script writes the processor model and the raw measurements (see
--out); the sizes in the model (cores, L1, L2 less alto's reserve) are the SDK's.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PWORD = 64
MULS = 32
CHAINS = 8
COPY_PWORDS = 8192  # 2 MiB of float32
L2_REPEATS = 8
L1_REPEATS = 4
ROWS, PWORDS = 128, 16  # the DPU kernels' input, 128 x 1024 float32
MVM_M = 512
# Weights K x N filling 256 IMC blocks of 64 x 64: square, four times as deep as wide (as many bytes stream in as out),
# and deeper.
MVM_SHAPES = [(1024, 1024), (2048, 512), (4096, 256)]
MVM_REPEATS = 8


@dataclass(frozen=True)
class Generation:
    name: str
    alto: str  # alto's HardwareGeneration
    clock_hz: (
        float  # the AI cores' clock, by which the profiler's cycles convert to time
    )


GENERATIONS = {
    "metis": Generation("metis", "OMEGA", 800e6),
    "europa": Generation("europa", "EUROPA", 1200e6),
}


def _sdk_root() -> Path:
    """The Voyager SDK checkout: ``VOYAGER_SDK``, or the one whose virtual environment runs Python."""
    root = Path(os.environ.get("VOYAGER_SDK", Path(sys.prefix).parent))
    if not (root / "tests" / "integration" / "alto").is_dir():
        raise RuntimeError(f"no Voyager SDK checkout at {root}; set VOYAGER_SDK")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class Card:
    """Compiles AxScript modules with alto for one generation and times them on its card."""

    def __init__(self, generation: Generation, compile_only: bool, iterations: int):
        self.sdk = _sdk_root()
        from axelera.compiler.alto.compiler import device
        from axelera.compiler.alto.compiler.config import HardwareGeneration

        self.generation = generation
        self.alto_generation = HardwareGeneration[generation.alto]
        self.compile_only = compile_only
        self.iterations = iterations
        self.cores = device.NUM_AI_CORES[self.alto_generation]
        sizes = {
            str(scope): size
            for scope, size in device.MEMORY_SIZES[self.alto_generation].items()
        }
        self.l1_bytes = sizes["global.l1"]
        self.l2_bytes = sizes["global.l2"]
        self.l2_reserved = device.L2_RESERVED_SIZE[self.alto_generation]
        self.imc_blocks = device.IMC_SIZE[self.alto_generation]

    def _config(self, out_dir: Path, cores: int):
        from axelera.compiler.alto.compiler.config import AltoConfig, Target

        return AltoConfig(
            output_dir=out_dir,
            generation=self.alto_generation,
            target=Target.DEVICE,
            num_cores=cores,
            enable_tiling_injection=False,
            enable_llm_support=True,
            assign_unique_input_pools=True,
            assign_unique_output_pools=True,
            unpack_axm_archive=True,
            clean_output_dir=True,
            emit_atex_artifacts=True,
        )

    def run(self, source: str, name: str, inputs: list[np.ndarray], cores: int = 1):
        """The module's median cycles on the slowest core and its outputs; cycles None if the profiler collected
        nothing, and None for both with --compile-only."""
        module = _load(source, name)
        out_dir = Path(tempfile.mkdtemp(prefix=f"calibrate_{name}_"))
        config = self._config(out_dir, cores)
        if self.compile_only:
            from axelera.compiler.alto.compiler.driver import compile

            compile(module, config)
            print(f"  {name}: compiled", flush=True)
            return None, None
        from axelera.compiler.alto.compiler.config import (
            AltoRuntimeConfig,
            ProfilingLevel,
        )
        from tests.integration.alto.utils import compile_and_run

        runtime = AltoRuntimeConfig(
            profiling_levels=[ProfilingLevel.DEVICE_MAIN],
            iterations=self.iterations,
            profiling_output_dir=out_dir,
        )
        # The profiler now and then collects no stats from a run that worked; a second run does, once the first has
        # released the device.
        for attempt in range(2):
            try:
                # The profiler records the git commit of the working directory: the SDK checkout's.
                with _working_directory(self.sdk):
                    _, outputs = compile_and_run(
                        module, config, config_runtime=runtime, inputs=inputs
                    )
                break
            except AssertionError:
                if attempt == 1:
                    raise
            gc.collect()
            time.sleep(5)
        cycles = _median_cycles(out_dir)
        if cycles is None:
            print(f"  {name}: no samples", flush=True)
        else:
            print(
                f"  {name}: {cycles:.0f} cycles, {self.us(cycles):.1f} us", flush=True
            )
        return cycles, [np.asarray(output) for output in outputs]

    def us(self, cycles: float) -> float:
        return cycles / self.generation.clock_hz * 1e6

    def core_counts(self) -> list[int]:
        counts, count = [], 1
        while count <= self.cores:
            counts.append(count)
            count *= 2
        return counts


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _median_cycles(out_dir: Path) -> float | None:
    """The median cycles of the slowest core: each core counts its own instruction stream."""
    per_core = []
    for path in Path(out_dir).glob("perf_stats_*core_*.json"):
        device_main = json.loads(path.read_text())["perf_stats"]["device_main"]
        if "median_cycles" in device_main:
            per_core.append(device_main["median_cycles"])
    return max(per_core) if per_core else None


def _load(source: str, name: str):
    path = Path(tempfile.mkdtemp(prefix="calibrate_")) / f"{name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.M


def _dpu_module(chains: int, muls: int):
    """x * 2 per row, in L1, with ``muls`` more multiplications per PWORD in ``chains`` independent chains."""
    registers = 2 + max(chains, 1)
    body = [
        'v[0] = A.cast(s.pop(), "float18x64")',
    ]
    for chain in range(chains):
        body.append(f"v[{2 + chain}] = v[0] * v[1]")
    for step in range(muls // max(chains, 1) - 1 if chains else 0):
        for chain in range(chains):
            body.append(f"v[{2 + chain}] = v[{2 + chain}] * v[1]")
    for chain in range(1, chains):
        body.append(f"v[2] = v[2] + v[{2 + chain}]")
    body.append("v[0] = v[0] * v[1]" if not chains else "v[0] = v[2]")
    body.append("o.push(v[0])")
    dpu = "\n".join(" " * 28 + line for line in body)
    return f"""import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
_c_lanes = [2.0] * 64
@A.ir_module
class M:
    @A.prim_func
    def main(x: A.Buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.DDR)
        c = A.alloc_constant(_c_lanes, "float32x64", scope=A.DDR)
        with A.AICore(core=0):
            li = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.L1)
            lo = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.L1)
            atomic.copy(x, scope=A.L1, output=li)
            for r in A.grid({ROWS}):
                with A.Datapath():
                    s = A.decl_stream("float32x64")
                    o = A.decl_stream("float32x64")
                    with A.IFD(num=0):
                        for p in A.serial({PWORDS}):
                            s.push(li[r, p])
                    with A.DPU():
                        v = A.array("float18x64", size={registers})
                        v[1] = A.cast(c[0], "float18x64")
                        for p in A.serial({PWORDS}):
{dpu}
                    with A.ODR():
                        for p in A.serial({PWORDS}):
                            lo[r, p] = o.pop()
            atomic.copy(lo, scope=A.DDR, output=out)
        A.output(out)
"""


def _ddr_module(cores: int):
    """Each of ``cores`` cores copies its own 2 MiB from DDR into its L1, and one PWORD back."""
    per_core = "\n".join(
        f"""        with A.AICore(core={core}):
            l = A.alloc_buffer(({COPY_PWORDS},), "float32x64", scope=A.L1)
            atomic.copy(x[{core}, A.Ramp(0, 1, {COPY_PWORDS})], scope=A.L1, output=l)
            atomic.copy(l[A.Ramp(0, 1, 1)], scope=A.DDR, output=out[A.Ramp({core}, 1, 1)])"""
        for core in range(cores)
    )
    return f"""import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
@A.ir_module
class M:
    @A.prim_func
    def main(x: A.Buffer(({cores}, {COPY_PWORDS}), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer(({cores},), "float32x64", scope=A.DDR)
{per_core}
        A.output(out)
"""


def _l2_module(repeats: int, cores: int = 1):
    """One DDR -> L2 copy of 2 MiB on core 0, then on each of ``cores`` cores ``repeats`` L2 -> L1 copies of it."""
    copies = "\n".join(
        " " * 12 + "atomic.copy(l2, scope=A.L1, output=l)" for _ in range(repeats)
    )
    per_core = "\n".join(
        f"""        with A.AICore(core={core}):
            l = A.alloc_buffer(({COPY_PWORDS},), "float32x64", scope=A.L1)
{"            atomic.copy(x, scope=A.L2, output=l2)" if core == 0 else ""}
            A.barrier({", ".join(str(c) for c in range(cores))})
{copies}
            atomic.copy(l[A.Ramp(0, 1, 1)], scope=A.DDR, output=out[A.Ramp({core}, 1, 1)])"""
        for core in range(cores)
    )
    return f"""import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
@A.ir_module
class M:
    @A.prim_func
    def main(x: A.Buffer(({COPY_PWORDS},), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer(({cores},), "float32x64", scope=A.DDR)
        l2 = A.alloc_buffer(({COPY_PWORDS},), "float32x64", scope=A.L2)
{per_core}
        A.output(out)
"""


def _l1_module(passes: int):
    """a + b over 128 x 1024 float32 in L1, ``passes`` times: two IFDs and an ODR streaming through L1."""
    datapath = f"""            for r in A.grid({ROWS}):
                with A.Datapath():
                    s = A.decl_stream("float32x64")
                    t = A.decl_stream("float32x64")
                    o = A.decl_stream("float32x64")
                    with A.IFD(num=0):
                        for p in A.serial({PWORDS}):
                            s.push(la[r, p])
                    with A.IFD(num=1):
                        for p in A.serial({PWORDS}):
                            t.push(lb[r, p])
                    with A.DPU():
                        for p in A.serial({PWORDS}):
                            o.push(s.pop() + t.pop())
                    with A.ODR():
                        for p in A.serial({PWORDS}):
                            lo[r, p] = o.pop()"""
    body = "\n".join([datapath] * passes)
    return f"""import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
@A.ir_module
class M:
    @A.prim_func
    def main(a: A.Buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.DDR), b: A.Buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.DDR)
        with A.AICore(core=0):
            la = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.L1)
            lb = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.L1)
            lo = A.alloc_buffer(({ROWS}, {PWORDS}), "float32x64", scope=A.L1)
            atomic.copy(a, scope=A.L1, output=la)
            atomic.copy(b, scope=A.L1, output=lb)
{body}
            atomic.copy(lo, scope=A.DDR, output=out)
        A.output(out)
"""


def _mvm_module(repeats: int, k: int, n: int):
    """Weights of K x N int8 loaded into the IMC once; then ``repeats`` matmuls of M = 512 rows."""
    k_pwords, n_pwords = k // PWORD, n // PWORD
    matmuls = "\n".join(
        " " * 12 + "atomic.matmul(matrix_ifd=la, matrix_imc=imc, output=lo)"
        for _ in range(repeats)
    )
    return f"""import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
@A.ir_module
class M:
    @A.prim_func
    def main(a: A.Buffer(({MVM_M}, {k_pwords}), "int8x64", scope=A.DDR), b: A.Buffer(({k}, {n_pwords}), "int8x64", scope=A.DDR)):
        out = A.alloc_buffer(({MVM_M}, {n_pwords}), "float32x64", scope=A.DDR)
        with A.AICore(core=0):
            la = A.alloc_buffer(({MVM_M}, {k_pwords}), "int8x64", scope=A.L1)
            lb = A.alloc_buffer(({k}, {n_pwords}), "int8x64", scope=A.L1)
            lo = A.alloc_buffer(({MVM_M}, {n_pwords}), "float32x64", scope=A.L1)
            imc = A.alloc_imc(({k_pwords}, {n_pwords}))
            atomic.copy(a, scope=A.L1, output=la)
            atomic.copy(b, scope=A.L1, output=lb)
            atomic.load_imc(buffer=lb, imc=imc)
{matmuls}
            atomic.copy(lo, scope=A.DDR, output=out)
        A.output(out)
"""


def _launch_module():
    """One PWORD from DDR into L1 and back on core 0: what a module costs beyond its work."""
    return """import axelera.compiler.alto.script as A
from axelera.compiler.alto.axol import atomic
@A.ir_module
class M:
    @A.prim_func
    def main(x: A.Buffer((1,), "float32x64", scope=A.DDR)):
        out = A.alloc_buffer((1,), "float32x64", scope=A.DDR)
        with A.AICore(core=0):
            l = A.alloc_buffer((1,), "float32x64", scope=A.L1)
            atomic.copy(x, scope=A.L1, output=l)
            atomic.copy(l, scope=A.DDR, output=out)
        A.output(out)
"""


def _less(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def dpu(card: Card) -> dict:
    x = (
        np.random.default_rng(0)
        .standard_normal((ROWS, PWORDS, PWORD))
        .astype(np.float32)
    )
    base, _ = card.run(_dpu_module(0, 0), "dpu_base", [x])
    result = {}
    for chains, key in ((1, "dependent"), (CHAINS, "independent")):
        cycles, _ = card.run(_dpu_module(chains, MULS), f"dpu_{chains}_chains", [x])
        # Beyond the base's one multiplication: the chains' multiplications, the additions joining them and a move.
        instructions = ROWS * PWORDS * (MULS + chains - 1)
        extra = _less(cycles, base)
        result[f"cycles_per_instruction_{key}"] = (
            None if extra is None else extra / instructions
        )
    return result


def mvm(card: Card) -> dict:
    rng = np.random.default_rng(0)
    shapes = []
    for k, n in MVM_SHAPES:
        a = rng.integers(-1, 2, (MVM_M, k), dtype=np.int8)
        b = rng.integers(-1, 2, (k, n), dtype=np.int8)
        inputs = [a.reshape(MVM_M, k // PWORD, PWORD), b.reshape(k, n // PWORD, PWORD)]
        once, outputs = card.run(_mvm_module(1, k, n), f"mvm_once_{k}x{n}", inputs)
        correct = None
        if outputs is not None:
            # Sums of at most 4096 products of -1, 0 and 1 are exact in float18.
            expected = (a.astype(np.int32) @ b.astype(np.int32)).astype(np.float32)
            correct = bool(np.array_equal(outputs[0].reshape(MVM_M, n), expected))
        repeated, _ = card.run(
            _mvm_module(1 + MVM_REPEATS, k, n), f"mvm_repeated_{k}x{n}", inputs
        )
        extra = _less(repeated, once)
        rows = MVM_REPEATS * MVM_M
        shapes.append(
            {
                "k": k,
                "n": n,
                "correct": correct,
                "cycles_per_row": None if extra is None else extra / rows,
                "macs_per_cycle": None if extra is None else rows * k * n / extra,
            }
        )
    return {"shapes": shapes}


def l1(card: Card) -> dict:
    rng = np.random.default_rng(0)
    a, b = (
        rng.standard_normal((ROWS, PWORDS, PWORD)).astype(np.float32) for _ in range(2)
    )
    once, _ = card.run(_l1_module(1), "l1_once", [a, b])
    repeated, _ = card.run(_l1_module(1 + L1_REPEATS), "l1_repeated", [a, b])
    extra = _less(repeated, once)
    moved = L1_REPEATS * 3 * ROWS * PWORDS * PWORD * 4
    return {"bytes_per_cycle_per_core": None if extra is None else moved / extra}


def l2(card: Card) -> dict:
    x = (
        np.random.default_rng(0)
        .standard_normal((COPY_PWORDS, PWORD))
        .astype(np.float32)
    )
    rates = {}
    for cores in card.core_counts():
        once, _ = card.run(
            _l2_module(1, cores), f"l2_once_{cores}_cores", [x], cores=cores
        )
        repeated, _ = card.run(
            _l2_module(1 + L2_REPEATS, cores),
            f"l2_repeated_{cores}_cores",
            [x],
            cores=cores,
        )
        extra = _less(repeated, once)
        moved = cores * L2_REPEATS * COPY_PWORDS * PWORD * 4
        rates[cores] = None if extra is None else moved / extra
    return {"bytes_per_cycle": rates}


def ddr(card: Card) -> dict:
    rates = {}
    for cores in card.core_counts():
        x = (
            np.random.default_rng(0)
            .standard_normal((cores, COPY_PWORDS, PWORD))
            .astype(np.float32)
        )
        cycles, outputs = card.run(
            _ddr_module(cores), f"ddr_{cores}_cores", [x], cores=cores
        )
        if outputs is not None and not np.array_equal(
            outputs[0].reshape(cores, PWORD), x[:, 0]
        ):
            raise RuntimeError(f"the DDR copies on {cores} cores returned wrong data")
        moved = cores * COPY_PWORDS * PWORD * 4
        rates[cores] = None if cycles is None else moved / cycles
    return {"bytes_per_cycle": rates}


def launch(card: Card) -> dict:
    x = np.zeros((1, PWORD), dtype=np.float32)
    cycles, _ = card.run(_launch_module(), "launch", [x])
    return {"us": None if cycles is None else card.us(cycles)}


KERNELS = {"dpu": dpu, "mvm": mvm, "l1": l1, "l2": l2, "ddr": ddr, "launch": launch}


def processor_model(card: Card, measured: dict) -> tuple[dict, list[str]]:
    """The processor model of the axelera target from the measurements, as {"axelera": {...}}, and the fields left
    0 for lack of a measurement."""
    mhz = card.generation.clock_hz / 1e6
    cores = card.cores
    missing = []

    def field(name: str, value: float | None) -> float:
        if value is None:
            missing.append(name)
            return 0.0
        return round(value, 1)

    cpi = measured.get("dpu", {}).get("cycles_per_instruction_independent")
    # The DPU issues at most one instruction per cycle; the difference of two runs can come out slightly below.
    fp32 = None if cpi is None else PWORD * mhz / max(cpi, 1.0) * cores
    macs = [
        s["macs_per_cycle"]
        for s in measured.get("mvm", {}).get("shapes", [])
        if s["macs_per_cycle"]
    ]
    int8 = 2 * max(macs) * mhz * cores if macs else None
    l1_rate = measured.get("l1", {}).get("bytes_per_cycle_per_core")
    l2_rates = {
        c: r for c, r in measured.get("l2", {}).get("bytes_per_cycle", {}).items() if r
    }
    # The L2 buses are per core: the rate per core on the most cores measured, on all of them.
    l2_bw = None
    if l2_rates:
        most = max(l2_rates)
        l2_bw = l2_rates[most] / most * cores * mhz
    ddr_rates = [
        r for r in measured.get("ddr", {}).get("bytes_per_cycle", {}).values() if r
    ]
    model = {
        "name": f"axelera_{card.generation.name}",
        "compute": {
            "int8": {"peak_flop": field("compute.int8", int8), "pipe": "mvm"},
            "fp32": {"peak_flop": field("compute.fp32", fp32), "pipe": "dpu"},
        },
        "memory": {
            "cache_line_bytes": 64,
            "scratchpad": [
                {"name": "l2", "size_bytes": card.l2_bytes - card.l2_reserved},
                {"name": "l1", "size_bytes": card.l1_bytes, "owner": "device"},
            ],
            "levels": [
                {
                    "name": "l1",
                    "size_bytes": card.l1_bytes,
                    "bandwidth": field(
                        "l1", None if l1_rate is None else l1_rate * mhz * cores
                    ),
                },
                {
                    "name": "l2",
                    "size_bytes": card.l2_bytes,
                    "bandwidth": field("l2", l2_bw),
                },
                {
                    "name": "dram",
                    "bandwidth": field(
                        "dram", max(ddr_rates) * mhz if ddr_rates else None
                    ),
                },
            ],
        },
        "concurrency": {
            "levels": [
                {"name": "device", "width": cores},
                {"name": "simd", "width": PWORD},
            ]
        },
        "launch_overhead": field(
            "launch_overhead", measured.get("launch", {}).get("us")
        ),
        "metrics": {},
    }
    return {"axelera": model}, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "kernels",
        nargs="*",
        help=f"kernels to run, of {', '.join(KERNELS)}; all by default",
    )
    parser.add_argument(
        "--arch",
        choices=list(GENERATIONS),
        default="metis",
        help="the card's generation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("."),
        help="folder for the model and measurements",
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="profiled runs per module"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="compile every kernel without running it",
    )
    args = parser.parse_args()
    unknown = [name for name in args.kernels if name not in KERNELS]
    if unknown:
        parser.error(
            f"unknown kernels {', '.join(unknown)}; choose from {', '.join(KERNELS)}"
        )

    args.out = args.out.resolve()
    card = Card(GENERATIONS[args.arch], args.compile_only, args.iterations)
    print(
        f"{args.arch}: {card.cores} AI cores at {card.generation.clock_hz / 1e6:.0f} MHz, "
        f"L1 {card.l1_bytes >> 20} MiB, L2 {card.l2_bytes >> 20} MiB, IMC {card.imc_blocks} blocks",
        flush=True,
    )
    measured, failed = {}, {}
    for name in args.kernels or KERNELS:
        print(name, flush=True)
        try:
            measured[name] = KERNELS[name](card)
        except (
            Exception
        ):  # noqa: BLE001 -- one kernel failing should not lose the others
            failed[name] = traceback.format_exc()
            print(failed[name], flush=True)
    if args.compile_only:
        print(
            "compiled: "
            + ", ".join(measured)
            + ("; failed: " + ", ".join(failed) if failed else "")
        )
        return 1 if failed else 0

    model, missing = processor_model(card, measured)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / f"axelera-{args.arch}.json"
    raw_path = args.out / f"axelera-{args.arch}.measurements.json"
    model_path.write_text(json.dumps(model, indent=4) + "\n")
    raw = {
        "generation": args.arch,
        "clock_hz": card.generation.clock_hz,
        "measured": measured,
        "failed": failed,
    }
    raw_path.write_text(json.dumps(raw, indent=4, default=str) + "\n")
    print(json.dumps(measured, indent=2, default=str))
    print(f"wrote {model_path} and {raw_path}")
    if missing:
        print("not measured (0 in the model): " + ", ".join(missing))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
