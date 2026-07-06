"""Pytest discovery/config for the pylulesh per-function tests.

The pylulesh modules (``lulesh``, ``domain``, ``util``, ``constants``) use bare
top-level imports, so the pylulesh directory must be on ``sys.path`` for the
tests to import them.
"""

import os
import sys

_PYLULESH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYLULESH_DIR not in sys.path:
    sys.path.insert(0, _PYLULESH_DIR)
