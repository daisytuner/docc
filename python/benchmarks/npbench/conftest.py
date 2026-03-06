import pytest
from benchmarks.npbench import harness


@pytest.fixture(autouse=True)
def capture_capsys_global(capsys):
    """
    Automatically capture the capsys fixture for every test and
    store it in the harness module global variable.
    This allows SDFGVerification to use capsys.disabled() without
    requiring it to be passed explicitly from every test function.
    """
    harness._GLOBAL_CAPSYS = capsys
    yield
    harness._GLOBAL_CAPSYS = None
