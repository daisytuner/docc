import pytest
import docc.python


@pytest.fixture(autouse=True, scope="function")
def clear_global_state():
    yield None
    docc.python.reset_target_registry()
