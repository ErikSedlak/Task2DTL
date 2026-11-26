"""Compile docs/report/report.typ to report.pdf (run: uv run --with typst python docs/report/build.py)."""
from pathlib import Path

import typst

here = Path(__file__).resolve().parent
typst.compile(str(here / "report.typ"), output=str(here / "report.pdf"), root=str(here.parent))
print("wrote", here / "report.pdf")
