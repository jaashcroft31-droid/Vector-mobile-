"""
Demo-app entrypoint - runs the exact same code as WA.py.

Streamlit Community Cloud ties one deployed app to one specific
(repository, branch, file) combination, so a second app from this same
repo/branch needs its own, different main file - it won't let you deploy
the exact same file twice, it just takes you to the existing app instead.

Rather than maintaining a second copy of the app (which would drift out
of sync every time WA.py changes), this file just executes WA.py's
current content every time it runs. It should never need to change again
- point Streamlit's "Main file path" at THIS file for the demo deployment,
keep using WA.py for your personal one, and every future change to WA.py
applies to both automatically.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "WA.py"), run_name="__main__")