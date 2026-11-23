"""Write a copy of the tutorial notebook without its explanatory text, for presenting.

    python3 presentation.py          # or: make presentation

The notebook keeps two kinds of markdown cells.  Headings, figures, equations and tables
are always shown.  The prose that explains them sits in separate cells tagged "reading":
a reader running the notebook alone wants it, an audience watching a talk does not.
This script drops the "reading" cells and writes mps_cfd_building_blocks_presentation.ipynb
next to the original, outputs included.  The original is not changed.
"""

import json
from pathlib import Path

SOURCE = Path(__file__).with_name("mps_cfd_building_blocks.ipynb")
TARGET = SOURCE.with_name(SOURCE.stem + "_presentation.ipynb")
HIDDEN_TAG = "reading"


def main():
    notebook = json.loads(SOURCE.read_text())
    cells = notebook["cells"]
    notebook["cells"] = [c for c in cells if HIDDEN_TAG not in c["metadata"].get("tags", [])]
    TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    print(f"{TARGET.name}: {len(notebook['cells'])} of {len(cells)} cells")


if __name__ == "__main__":
    main()
