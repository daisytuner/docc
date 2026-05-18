import pytest
import subprocess
import os

from pathlib import Path

# This method clones / fetches the llvm-test-suite repository
@pytest.fixture(scope="session")
def setup():
    # The commit sha on which the llvm-test-suite is fixed
    COMMIT = "f711e105d94c4819d3bc8f399f06f22d4df49421"


    # Check the repository dir
    repo_dir = Path(__file__).parent / "llvm-test-suite"
    if repo_dir.exists():
        # The repository already exists, check that its a folder
        assert repo_dir.is_dir(), "The repository path already exists but is not a directory: " + str(repo_dir)
        assert (repo_dir / ".git").is_dir(), "The repository dir already exists but is not a git repository: " + str(repo_dir)
        # Fetch all
        fetch_process = subprocess.Popen(
            ["git", "fetch", "-q", "--all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            cwd=repo_dir,
        )
        stdout, stderr = fetch_process.communicate()
        assert fetch_process.returncode == 0, "Could not fetch the llvm-test-suite repository"
    else:
        # The repository does not exist
        # We need to clone it
        clone_process = subprocess.Popen(
            ["git", "clone", "-q", "https://github.com/llvm/llvm-test-suite.git", str(repo_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = clone_process.communicate()
        assert clone_process.returncode == 0, "Could not clone the llvm-test-suite repository"

    # Now, we have to checkout the specified branch / commit
    checkout_process = subprocess.Popen(
        ["git", "checkout", "-q", COMMIT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=repo_dir,
    )
    stdout, stderr = checkout_process.communicate()
    assert checkout_process.returncode == 0, "Could not checkout the llvm-test-suite repository to commit: " + COMMIT

    # Check the build dir
    build_dir = repo_dir / "build"
    if build_dir.exists():
        assert build_dir.is_dir(), "The build path already exists but is not a directory: " + str(build_dir)
    else:
        # Create the buil dir
        os.mkdir(str(build_dir))

    # Always configure CMake for good measure
    cmake_process = subprocess.Popen(
        [
            "cmake",
            "-DCMAKE_C_COMPILER=docc",
            "-DCMAKE_CXX_COMPILER=docc-cpp",
            "-DTEST_SUITE_BENCHMARKING_ONLY=ON",
            "-DTEST_SUITE_COLLECT_CODE_SIZE=OFF",
            "-C", "../cmake/caches/O2.cmake",
            "-DTEST_SUITE_SUBDIRS=SingleSource;MultiSource",
            str(repo_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=build_dir,
    )
    stdout, stderr = cmake_process.communicate()
    if cmake_process.returncode != 0:
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
    assert cmake_process.returncode == 0, "CMake configuration failed"

    # Clean all build files
    make_clean_process = subprocess.Popen(
        ["make", "clean"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=build_dir,
    )
    stdout, stderr = make_clean_process.communicate()
    if make_clean_process.returncode != 0:
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
    assert make_clean_process.returncode == 0, "make clean failed"

    yield repo_dir, build_dir

# Each test is listed in the parameters
# Options for compiles:
#   YES = The test compiles
#   TIMEOUT = The compilation timeouts (5 min)
#   OUT_OF_MEMORY = The compiler's memory usage crashes the system
#   SEGFAULT = The compiler segfaults
# Options for executes:
#   PASS = The test execution passes
#   TIMEOUT = The test execution timeouts (5 min)
#   FAIL = The test execution fails because the result is wrong or the application crashes
#   FLAKY = The test execution sometimes passes, sometimes fails
@pytest.mark.parametrize(
    "path, name, compiles, executes",
    [
        pytest.param("MultiSource/Applications/aha", "aha", "YES", "PASS"),
        pytest.param("MultiSource/Applications/ALAC/decode", "alacconvert-decode", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/ALAC/encode", "alacconvert-encode", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/ClamAV", "clamscan", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/d", "make_dparser", "TIMEOUT", ""),
        pytest.param("MultiSource/Applications/hbd", "hbd", "YES", "PASS"),
        pytest.param("MultiSource/Applications/hexxagon", "hexxagon", "YES", "TIMEOUT"),
        pytest.param("MultiSource/Applications/JM/ldecod", "ldecod", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/JM/lencod", "lencod", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/kimwitu++", "kc", "TIMEOUT", ""),
        pytest.param("MultiSource/Applications/lambda-0.1.3", "lambda", "YES", "PASS"),
        pytest.param("MultiSource/Applications/lua", "lua", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/minisat", "minisat", "YES", "PASS"),
        pytest.param("MultiSource/Applications/obsequi", "Obsequi", "YES", "PASS"),
        pytest.param("MultiSource/Applications/oggenc", "oggenc", "YES", "FAIL"),
        pytest.param("MultiSource/Applications/sgefa", "sgefa", "YES", "TIMEOUT"),
        pytest.param("MultiSource/Applications/SIBsim4", "SIBsim4", "TIMEOUT", ""),
        pytest.param("MultiSource/Applications/siod", "siod", "TIMEOUT", ""),
        pytest.param("MultiSource/Applications/SPASS", "SPASS", "SEGFAULT", ""),
        pytest.param("MultiSource/Applications/spiff", "spiff", "SEGFAULT", ""),
        pytest.param("MultiSource/Applications/sqlite3", "sqlite3", "TIMEOUT", ""),
        pytest.param("MultiSource/Applications/viterbi", "viterbi", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/7zip", "7zip-benchmark", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/ASC_Sequoia/AMGmk", "AMGmk", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/ASC_Sequoia/CrystalMk", "CrystalMk", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/ASC_Sequoia/IRSmk", "IRSmk", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/ASCI_Purple/SMG2000", "smg2000", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/BitBench/drop3", "drop3", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/BitBench/five11", "five11", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/BitBench/uudecode", "uudecode", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/BitBench/uuencode", "uuencode", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Bullet", "bullet", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/CoMD", "CoMD", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/miniAMR", "miniAMR", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/miniGMG", "miniGMG", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/Pathfinder", "PathFinder", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/RSBench", "rsbench", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/SimpleMOC", "SimpleMOC", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C/XSBench", "XSBench", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C++/CLAMR", "CLAMR", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C++/HACCKernels", "HACCKernels", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C++/HPCCG", "HPCCG", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C++/miniFE", "miniFE", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/DOE-ProxyApps-C++/PENNANT", "PENNANT", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Fhourstones", "fhourstones", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Fhourstones-3.1", "fhourstones3.1", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/analyzer", "analyzer", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/distray", "distray", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/fourinarow", "fourinarow", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/mason", "mason", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/neural", "neural", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/pcompress2", "pcompress2", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/FreeBench/pifft", "pifft", "YES", "TIMEOUT"),
        pytest.param("MultiSource/Benchmarks/llubenchmark", "llu", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/mafft", "pairlocalalign", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/MallocBench/cfrac", "cfrac", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/MallocBench/espresso", "espresso", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/MallocBench/gs", "gs", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/McCat/01-qbsort", "qbsort", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/McCat/03-testtrie", "testtrie", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/McCat/04-bisect", "bisect", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/McCat/05-eks", "eks", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/McCat/08-main", "main", "SEGFAULT", ""),
        pytest.param("MultiSource/Benchmarks/McCat/09-vor", "vor", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/McCat/12-IOtest", "iotest", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/McCat/17-bintr", "bintr", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/McCat/18-imp", "imp", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/mediabench/adpcm/rawcaudio", "rawcaudio", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/mediabench/adpcm/rawdaudio", "rawdaudio", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/mediabench/g721/g721encode", "encode", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/mediabench/gsm/toast", "toast", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/mediabench/jpeg/jpeg-6a", "cjpeg", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/mediabench/mpeg2/mpeg2dec", "mpeg2decode", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/MiBench/automotive-basicmath", "automotive-basicmath", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/automotive-bitcount", "automotive-bitcount", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/automotive-susan", "automotive-susan", "OUT_OF_MEMORY", ""),
        pytest.param("MultiSource/Benchmarks/MiBench/consumer-jpeg", "consumer-jpeg", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/MiBench/consumer-lame", "consumer-lame", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/consumer-typeset", "consumer-typeset", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/MiBench/network-dijkstra", "network-dijkstra", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/MiBench/network-patricia", "network-patricia", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/security-rijndael", "security-rijndael", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/security-sha", "security-sha", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/MiBench/telecomm-CRC32", "telecomm-CRC32", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/telecomm-FFT", "telecomm-fft", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/MiBench/telecomm-gsm", "telecomm-gsm", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/nbench", "nbench", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/NPB-serial/is", "is", "YES", "FLAKY"),
        pytest.param("MultiSource/Benchmarks/Olden/bh", "bh", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/bisort", "bisort", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Olden/em3d", "em3d", "YES", "TIMEOUT"),
        pytest.param("MultiSource/Benchmarks/Olden/health", "health", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/mst", "mst", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/perimeter", "perimeter", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/power", "power", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/treeadd", "treeadd", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Olden/tsp", "tsp", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Olden/voronoi", "voronoi", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/PAQ8p", "paq8p", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C/agrep", "agrep", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/Prolangs-C/bison", "mybison", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C/gnugo", "gnugo", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/city", "city", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/employ", "employ", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/life", "life", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/ocean", "ocean", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/primes", "primes", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Prolangs-C++/simul", "simul", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Ptrdist/anagram", "anagram", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Ptrdist/bc", "bc", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/Ptrdist/ft", "ft", "YES", "TIMEOUT"),
        pytest.param("MultiSource/Benchmarks/Ptrdist/ks", "ks", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Ptrdist/yacr2", "yacr2", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Rodinia/backprop", "backprop", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Rodinia/hotspot", "hotspot", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Rodinia/pathfinder", "pathfinder", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Rodinia/srad", "srad", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/SciMark2-C", "scimark2", "YES", "FAIL"),
        pytest.param("MultiSource/Benchmarks/sim", "sim", "SEGFAULT", ""),
        pytest.param("MultiSource/Benchmarks/tramp3d-v4", "tramp3d-v4", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/Trimaran/enc-3des", "enc-3des", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Trimaran/enc-md5", "enc-md5", "TIMEOUT", ""),
        pytest.param("MultiSource/Benchmarks/Trimaran/enc-pc1", "enc-pc1", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Trimaran/enc-rc4", "enc-rc4", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Trimaran/netbench-crc", "netbench-crc", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/Trimaran/netbench-url", "netbench-url", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/ControlFlow-dbl", "ControlFlow-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/ControlFlow-flt", "ControlFlow-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/ControlLoops-dbl", "ControlLoops-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/ControlLoops-flt", "ControlLoops-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/CrossingThresholds-dbl", "CrossingThresholds-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/CrossingThresholds-flt", "CrossingThresholds-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Equivalencing-dbl", "Equivalencing-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Equivalencing-flt", "Equivalencing-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Expansion-dbl", "Expansion-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Expansion-flt", "Expansion-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/GlobalDataFlow-dbl", "GlobalDataFlow-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/GlobalDataFlow-flt", "GlobalDataFlow-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/IndirectAddressing-dbl", "IndirectAddressing-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/IndirectAddressing-flt", "IndirectAddressing-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/InductionVariable-dbl", "InductionVariable-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/InductionVariable-flt", "InductionVariable-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LinearDependence-dbl", "LinearDependence-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LinearDependence-flt", "LinearDependence-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LoopRerolling-dbl", "LoopRerolling-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LoopRerolling-flt", "LoopRerolling-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LoopRestructuring-dbl", "LoopRestructuring-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/LoopRestructuring-flt", "LoopRestructuring-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/NodeSplitting-dbl", "NodeSplitting-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/NodeSplitting-flt", "NodeSplitting-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Packing-dbl", "Packing-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Packing-flt", "Packing-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Recurrences-dbl", "Recurrences-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Recurrences-flt", "Recurrences-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Reductions-dbl", "Reductions-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Reductions-flt", "Reductions-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Searching-dbl", "Searching-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Searching-flt", "Searching-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/StatementReordering-dbl", "StatementReordering-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/StatementReordering-flt", "StatementReordering-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Symbolics-dbl", "Symbolics-dbl", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/TSVC/Symbolics-flt", "Symbolics-flt", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/VersaBench/8b10b", "8b10b", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/VersaBench/beamformer", "beamformer", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/VersaBench/bmm", "bmm", "YES", "PASS"),
        pytest.param("MultiSource/Benchmarks/VersaBench/dbms", "dbms", "SEGFAULT", ""),
        pytest.param("MultiSource/Benchmarks/VersaBench/ecbdes", "ecbdes", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "functionobjects", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "loop_unroll", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "simple_types_constant_folding", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "simple_types_loop_invariant", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "stepanov_abstraction", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/Adobe-C++", "stepanov_vector", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame/Large", "fasta", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "fannkuch", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "n-body", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "nsieve-bits", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "partialsums", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "puzzle", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "recursive", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/BenchmarkGame", "spectral-norm", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/CoyoteBench", "almabench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/CoyoteBench", "fftbench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/CoyoteBench", "huffbench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/CoyoteBench", "lpbench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Dhrystone", "dry", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Dhrystone", "fldry", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Linpack", "linpack-pc", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/McGill", "chomp", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/McGill", "misr", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/McGill", "queens", "TIMEOUT", ""),
        pytest.param("SingleSource/Benchmarks/Misc", "dt", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "evalloop", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "fbench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "ffbench", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-1", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-3", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-4", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-5", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-6", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-7", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops-8", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "flops", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "fp-convert", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "himenobmtxpa", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "lowercase", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "mandel-2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "mandel", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "matmul_f64_4x4", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "oourafft", "YES", "FAIL"),
        pytest.param("SingleSource/Benchmarks/Misc", "perlin", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "pi", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "ReedSolomon", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "revertBits", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "richards_benchmark", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "salsa20", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc", "whetstone", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++/Large", "ray", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++/Large", "sphereflake", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++", "bigfib", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++", "mandel-text", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++", "oopack_v1p8", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++", "stepanov_container", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++", "stepanov_v1p2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Misc-C++-EH", "spirit", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/datamining/correlation", "correlation", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/datamining/covariance", "covariance", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/gemver", "gemver", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/gesummv", "gesummv", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/symm", "symm", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/syr2k", "syr2k", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/syrk", "syrk", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/blas/trmm", "trmm", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/kernels/atax", "atax", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/kernels/bicg", "bicg", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/kernels/doitgen", "doitgen", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/kernels/mvt", "mvt", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/cholesky", "cholesky", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/durbin", "durbin", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/gramschmidt", "gramschmidt", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/lu", "lu", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/ludcmp", "ludcmp", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/linear-algebra/solvers/trisolv", "trisolv", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/medley/deriche", "deriche", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/medley/floyd-warshall", "floyd-warshall", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/medley/nussinov", "nussinov", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/adi", "adi", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/fdtd-2d", "fdtd-2d", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/heat-3d", "heat-3d", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/jacobi-1d", "jacobi-1d", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/jacobi-2d", "jacobi-2d", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Polybench/stencils/seidel-2d", "seidel-2d", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-ackermann", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-ary3", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-fib2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-hash", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-heapsort", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-lists", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-matrix", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-methcall", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-nestedloop", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-objinst", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-random", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-sieve", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout", "Shootout-strcat", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++/EH", "Shootout-C++-except", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-ackermann", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-ary", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-ary2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-ary3", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-fibo", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-hash", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-hash2", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-heapsort", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-lists", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-lists1", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-matrix", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-methcall", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-moments", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-nestedloop", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-objinst", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-random", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-sieve", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Shootout-C++", "Shootout-C++-strcat", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/SmallPT", "smallpt", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Bubblesort", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "FloatMM", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Oscar", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Perm", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Puzzle", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Queens", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Quicksort", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "RealMM", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Towers", "YES", "PASS"),
        pytest.param("SingleSource/Benchmarks/Stanford", "Treesort", "YES", "PASS"),
    ]
)
def test(setup, path, name, compiles, executes):
    repo_dir, build_dir = setup

    # Check that dir / file exist
    assert (repo_dir / path).is_dir(), "Test path does not exist: " + path
    test_file = build_dir / path / (name + ".test")
    assert test_file.is_file(), "Test file does not exist: " + str(test_file)

    # Determine if all test should be tried to execute
    all_tests = ("ALL" in os.environ)

    # Check that compiles and executes have valid values
    assert compiles in ["YES", "TIMEOUT", "OUT_OF_MEMORY", "SEGFAULT"], "compiles option must be YES, TIMEOUT, OUT_OF_MEMORY, or SEGFAULT"
    if compiles == "YES":
        assert executes in ["PASS", "TIMEOUT", "FAIL", "FLAKY"], "executes option must be PASS, TIMEOUT, FAIL, or FLAKY"

    # Skip
    if compiles == "OUT_OF_MEMORY":
        pytest.skip("Compilation requires too much memory")
    if not all_tests:
        if compiles == "TIMEOUT":
            pytest.skip("Compilation timeouts")
        elif compiles == "SEGFAULT":
            pytest.xfail("Compilation segfaults")
        if executes == "TIMEOUT":
            pytest.skip("Execution timeouts")
        elif executes == "FAIL":
            pytest.xfail("Execution fails")
        elif executes == "FLAKY":
            pytest.skip("Execution flaky")

    # Compile
    make_process = subprocess.Popen(
        ["make", "-j", str(os.cpu_count()), name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=build_dir,
    )
    try:
        stdout, stderr = make_process.communicate(timeout=300)
    except subprocess.TimeoutExpired: # must catch this otherwise subprocess is not killed
        make_process.kill()
        if compiles == "TIMEOUT":
            return # Expected this
        pytest.fail("Compilation timed out but expected compiles = " + compiles)
    if make_process.returncode != 0:
        if compiles == "SEGFAULT":
            return # Expected this
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
    assert make_process.returncode == 0, "Compilation failed but expected compiles = " + compiles

    # Execute
    lit_process = subprocess.Popen(
        ["lit", "-v", str(test_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        cwd=build_dir,
    )
    try:
        stdout, stderr = lit_process.communicate(timeout=300)
    except subprocess.TimeoutExpired: # must catch this otherwise subprocess is not killed
        lit_process.kill()
        if executes == "TIMEOUT":
            return # Expected this
        pytest.fail("Execution timed out but expected executes = " + executes)
    if lit_process.returncode != 0:
        if executes == "FAIL" or executes == "FLAKY":
            return # Expected this
        print("STDOUT:\n", stdout)
        print("STDERR:\n", stderr)
    assert lit_process.returncode == 0, "Execution failed but expected executes = " + executes