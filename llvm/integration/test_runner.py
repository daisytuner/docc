import os
import subprocess

from typing import Dict
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp 
import threading

class SDFGVerification:
    def __init__(self, verification: dict):
        self._verification = verification
    
    def _parse_opt_report(self, output_str: str) -> Dict[str, int]:
        opt_mode = False
        stats = {}
        for line in output_str.splitlines():
            line = line.strip()
            if line.startswith("DOCC Optimization Report Start:"):
                opt_mode = True
            elif line.startswith("DOCC Optimization Report End"):
                opt_mode = False
            elif opt_mode:
                key, val = line.split(": ")
                if key.strip() in ["source_language", "target_triple", "data_layout"]:
                    continue
                if key.strip() not in stats:
                    stats[key.strip()] = 0
                stats[key.strip()] += int(val.strip())

        return stats

    def verify(self, stderr: str) -> None:
        stats = self._parse_opt_report(stderr)
        print(stats)
        for key, val in self._verification.items():
            if key not in stats:
                assert val == 0, f"Key {key} not found in stats"
            else:
                assert stats[key] == val, f"Key {key} has value {stats[key]} but expected {val}"
        print("All verifications passed.")

class TransformationVerification:
    def __init__(self, transformations: dict):
        self._transformations = transformations

    def verify(self, output_dir, stderr: str = None) -> None:
        import json
        opt_report_files = list(Path(output_dir).rglob("sdfg_0.opt_report.json"))
        if not opt_report_files:
            raise FileNotFoundError(f"No sdfg_0.opt_report.json found in {output_dir}")
        for opt_report_path in opt_report_files:
            print(f"Using opt_report file at: {opt_report_path}")
            # Load the opt_report JSON file
            with open(opt_report_path, 'r') as f:
                opt_report = json.load(f)
            regions = opt_report.get("regions", [])
            if regions != []:
                break

        # Build a lookup: loopnest_index -> region
        loopnest_to_region = {region.get("loopnest_index"): region for region in regions}

        for transformation, params in self._transformations.items():
            # Check for new dict format with 'loop_nests' and 'max_tuned_loops'
            if isinstance(params, dict) and 'loop_nests' in params and 'tuned_loops' in params:
                loop_nests = params['loop_nests']
                tuned_loops = params['tuned_loops']
                # Check that transformation is present in all specified loop_nests
                for idx in loop_nests:
                    region = loopnest_to_region.get(idx)
                    assert region is not None, f"Region with loopnest_index {idx} not found."
                    transformations = region.get("transformations", {})
                    assert transformation in transformations, (
                        f"Transformation {transformation} not found in region with loopnest_index {idx}."
                    )
                # Count total number of times transformation was applied
                applied_count = 0
                for region in regions:
                    transformations = region.get("transformations", {})
                    if transformation in transformations:
                        # Check if 'applied' is True (if present)
                        t = transformations[transformation]
                        if isinstance(t, dict) and t.get('applied', True):
                            applied_count += 1
                        elif t is True:
                            applied_count += 1
                assert applied_count == tuned_loops, (
                    f"Transformation {transformation} applied {applied_count} times but expected {tuned_loops} times"
                )
            else:
                # Fallback to old behavior: params is a set of indices
                for idx in params:
                    region = loopnest_to_region.get(idx)
                    assert region is not None, f"Region with loopnest_index {idx} not found."
                    transformations = region.get("transformations", {})
                    assert transformation in transformations, (
                        f"Transformation {transformation} not found in region with loopnest_index {idx}."
                    )
    print("All transformation verifications passed.")

class TestRunner:
    __test__ = False

    def __init__(
            self,
            test_suite,
            test_case,
            compiler,
            reference_compiler,
            flags,
            mode,
            additional_source_files,
            evaluation_function,
            sdfg_verification : SDFGVerification = None,
            transformation_verification : TransformationVerification = None,
            docc_flags: list = [],
            output_dir = None
        ) -> None:
        self._test_suite = test_suite
        self._test_case = test_case
        self._compiler = compiler
        self._reference_compiler = reference_compiler
        self._flags = flags
        self._mode = mode
        self._additional_source_files = additional_source_files
        self._evaluation_function = evaluation_function
        self._sdfg_verification = sdfg_verification
        self._transformation_verification = transformation_verification
        self._docc_flags = docc_flags
        if output_dir is None:
            rootDir = Path(__file__).parent / "pytestOut" / (test_suite+"-"+mode)
        else:
            rootDir = Path(output_dir)
        out_dir = rootDir / test_case.name
        Path.mkdir(out_dir, parents=True, exist_ok=True)
        self.output_dir_ = Path(mkdtemp(prefix=datetime.now().strftime('%Y-%m-%d_%H-%M-%S_'), dir= out_dir))

    def run(self, timeout=600) -> None:
        exception_bucket = []
        def target():
            try:
                reference_file = self._compile(self._test_case, reference=True)
                assert reference_file.is_file()
                test_file = self._compile(self._test_case, False)
                assert test_file.is_file()
                self._evaluation_function(reference_file, test_file)
            except Exception as e:
                exception_bucket.append(e)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)

        if exception_bucket:
            raise exception_bucket[0]

        if t.is_alive():
            raise TimeoutError("Timeout reached")
        
        return

    def _compile(self, test_case, reference: bool) -> None:
        if reference:
            cmd = [self._reference_compiler]
        else:
            cmd = [self._compiler]

        cmd += [str(path.absolute()) for path in self._additional_source_files]
        cmd.append(str(test_case.absolute()))
        cmd += self._flags

        if reference:
            os.environ["OMPI_CC"] = "clang-19"
            os.environ["OMPI_CXX"] = "clang++-19"
            os.environ["MPICH_CC"] = "clang-19"
            os.environ["MPICH_CXX"] = "clang++-19"
            output_path = self.output_dir_ / "reference"
        else:
            os.environ["OMPI_CC"] = "docc"
            os.environ["OMPI_CXX"] = "docc-cpp"
            os.environ["MPICH_CC"] = "docc"
            os.environ["MPICH_CXX"] = "docc-cpp"
            os.environ["DOCC_OPT_REPORT"] = "1"
            output_path = self.output_dir_ / "test"
            cmd += ["-docc-work-dir=" + str(self.output_dir_ / "DOCC")]
            cmd += ["-docc-tune=" + self._mode]
            cmd += self._docc_flags
            cmd += ["-docc-dot-scheduled"]

        output_file = output_path / "a.out"
        cmd += ["-o", str(output_file.absolute())]
        print("\nBuild command for", "reference" if reference else "test", "executable:\n", " ".join(cmd))

        Path.mkdir(output_path, exist_ok=False)
        my_env = os.environ.copy()
        my_env["LLVM_ENABLE_BACKTRACES"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding="utf-8",
            errors="backslashreplace",
            env=my_env,
        )
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)

        if not reference and self._sdfg_verification is not None:
            self._sdfg_verification.verify(stderr)

        if not reference and self._transformation_verification is not None:
            self._transformation_verification.verify(self.output_dir_, stderr)

        return output_file
