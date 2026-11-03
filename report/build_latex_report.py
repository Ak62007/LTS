"""Build the final LaTeX PDF and a self-contained editable source archive."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from report.build_latex_assets import ROOT, create_tables, draw_actual_points

SOURCE = ROOT / "report/latex/ED409_LTS_Final_Report.tex"
DESTINATION = ROOT / "report"


def package_source() -> Path:
    """Bundle the source, numeric fragments, figures, and executed-code images."""
    source = SOURCE.read_text()
    figures = re.findall(r"\\plot(?:\[[^\]]+\])?\{([^}]+)\}", source)
    code_images = re.findall(r"\\codeimage\{([^}]+)\}", source)
    archive = DESTINATION / "ED409_LTS_LaTeX_Source.zip"
    entries = {
        SOURCE.name: SOURCE,
        "README.md": SOURCE.parent / "README.md",
        "REVIEW_LOG.md": SOURCE.parent / "REVIEW_LOG.md",
        **{
            f"generated/{path.name}": path
            for path in sorted((SOURCE.parent / "generated").glob("*"))
            if path.suffix in {".tex", ".json"}
        },
        **{f"figures/{name}": ROOT / "outputs/figures" / name for name in figures},
        **{f"code/{name}": ROOT / "report/assets/final" / name for name in code_images},
    }
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for relative_name, path in entries.items():
            bundle.write(path, relative_name)
    return archive


def main() -> None:
    """Rebuild only report assets and compile; do not rerun statistical fits."""
    if not shutil.which("tectonic"):
        raise SystemExit("Install Tectonic before compiling the LaTeX report.")
    draw_actual_points()
    create_tables()
    subprocess.run(
        ["tectonic", "--keep-logs", "--outdir", str(DESTINATION), str(SOURCE)],
        check=True,
        cwd=ROOT,
    )
    archive = package_source()
    print(f"PDF: {DESTINATION / (SOURCE.stem + '.pdf')}")
    print(f"Editable source: {archive}")


if __name__ == "__main__":
    main()
