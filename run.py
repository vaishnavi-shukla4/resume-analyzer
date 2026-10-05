"""Launch the Streamlit application. Application logic lives in src/app.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "src" / "app.py"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=str(project_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
