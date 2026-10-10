"""The reference query of SPEC §5.7: `tgi selftest` (see `tgi/selftest.py`).

    uv run python tests/e2e/reference_query.py --scenario tests/scenarios/reference.json
    uv run python tests/e2e/reference_query.py      # the server at MCP_BIND, MCP_TOKEN

The logic lives in `tgi.selftest` so that the installed `tgi selftest` has it too.
"""

import sys

from tgi.cli import app

if __name__ == "__main__":
    app(["selftest", *sys.argv[1:]])
