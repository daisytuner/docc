from docc.compiler import CompiledSDFG
from docc.python import native
import pytest
import os
import ctypes
import shutil
import getpass


def test_default_output_folder():
    @native
    def my_func(a):
        pass

    # Compile without specifying output folder
    compiled = my_func.compile(1)

    # Check if it is in a temp folder with docc prefix
    user = os.getenv("USER")
    if not user:
        user = getpass.getuser()
    assert compiled.lib_path.startswith(f"/tmp/{user}/DOCC/my_func-")
    assert os.path.exists(compiled.lib_path)

    # Cleanup handled by tempfile (mostly), but we can try to remove the parent dir if we want
    # For now just checking existence is enough


def test_compile_simple_scalar():
    @native
    def simple_scalar(a, b):
        pass

    # Load the library
    compiled_sdfg = simple_scalar.compile(1, 2)

    assert isinstance(compiled_sdfg, CompiledSDFG)
    assert os.path.exists(compiled_sdfg.lib_path)

    # Call the function
    compiled_sdfg(1, 2)

    # Cleanup
    shutil.rmtree(os.path.dirname(compiled_sdfg.lib_path))


def test_compile_return_scalar():
    @native
    def return_scalar(a, b) -> int:
        pass

    compiled_sdfg = return_scalar.compile(1, 2)

    assert isinstance(compiled_sdfg, CompiledSDFG)
    assert os.path.exists(compiled_sdfg.lib_path)

    # Call the function
    # Since the body is empty, the return value is undefined (garbage), but it should not crash
    res = compiled_sdfg(1, 2)
    assert isinstance(res, int)

    # Cleanup
    shutil.rmtree(os.path.dirname(compiled_sdfg.lib_path))


def test_compile_return_arg():
    @native
    def return_arg(a, b) -> int:
        return a

    compiled_sdfg = return_arg.compile(10, 20)

    assert isinstance(compiled_sdfg, CompiledSDFG)

    # Call the function
    res = compiled_sdfg(10, 20)
    assert res == 10

    res = compiled_sdfg(42, 20)
    assert res == 42

    # Cleanup
    shutil.rmtree(os.path.dirname(compiled_sdfg.lib_path))


def test_caching_sdfg_count():
    @native
    def cached_func(a):
        pass

    # First compilation (int)
    cached_func.compile(1)
    assert len(cached_func.cache) == 1

    # Same type, should use cache
    cached_func.compile(2)
    assert len(cached_func.cache) == 1

    # Different type (float), new compilation
    cached_func.compile(1.0)
    assert len(cached_func.cache) == 2

    # Cleanup
    if len(cached_func.cache) > 0:
        first_compiled = list(cached_func.cache.values())[0]
        if os.path.exists(os.path.dirname(first_compiled.lib_path)):
            shutil.rmtree(os.path.dirname(first_compiled.lib_path))


def test_reserved_cpp_names():
    # 'main' is a reserved name in C/C++ (entry point)
    # 'exit' is a standard library function

    @native
    def main(a, b) -> int:
        c = a + b
        return c

    compiled = main.compile(1, 2)
    assert compiled(1, 2) == 3
    shutil.rmtree(os.path.dirname(compiled.lib_path))

    @native
    def exit(a) -> int:
        return a

    compiled = exit.compile(10)
    assert compiled(10) == 10
    shutil.rmtree(os.path.dirname(compiled.lib_path))
