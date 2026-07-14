## dist_kit/tests/conftest.py ##
##
## Makes the dist_kit package importable no matter where pytest is invoked
## from (the kit is meant to be dropped into arbitrary projects), and
## registers the markers used by this suite.

import os
import sys

# parent of dist_kit/ → the project root the kit was dropped into
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: real training subprocess tests — require GPU, ~30-60 s each"
    )
    config.addinivalue_line(
        "markers", "integration: needs the HF cache (tokenizer/dataset) — skipped when not cached"
    )
