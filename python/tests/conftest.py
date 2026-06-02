import pytest


def pytest_collection_modifyitems(items):
    # Pytest applies some built-in markers automatically under the hood.
    # We ignore those so they don't count as "marked".
    builtin_markers = {
        "parametrize",
        "skip",
        "skipif",
        "xfail",
        "usefixtures",
        "filterwarnings",
    }

    for item in items:
        # Get all markers applied to this test (including from classes/modules)
        test_markers = {m.name for m in item.iter_markers()}
        custom_markers = test_markers - builtin_markers

        # If no custom markers exist, apply our automatically generated one
        if not custom_markers:
            item.add_marker(pytest.mark.unmarked)
