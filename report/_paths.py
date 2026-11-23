"""Where the report scripts find the solver, its results and the data.

Defaults follow the repository layout; override with environment variables
when the snapshots live elsewhere (e.g. on scratch storage).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
SOLVER = os.environ.get("QCFD_SOLVER", os.path.join(ROOT, "scripts"))       # chorin_projection.py, bundle_simulation.py
DATA = os.environ.get("QCFD_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("QCFD_RESULTS", os.path.join(DATA, "results"))     # <tag>_params.pfile, temp/ snapshots
IC_DIR = os.path.join(DATA, "initial_conditions")
MASK_DIR = os.path.join(DATA, "mps_masks")
REF = os.environ.get("QCFD_REFERENCE", os.path.join(DATA, "star_ccm"))      # C_D.csv, C_L.csv, star_t*.vtk
OUT = os.environ.get("QCFD_REPORT_DATA", os.path.join(DATA, "archive"))     # extractor outputs read by plots.py
# Full Star-CCM+ VTK archive (2.6 GB, not shipped); only needed for snapshot
# times that are not already cached as data/star_ccm/star_t*.vtk.
STAR_TAR = os.environ.get("QCFD_STAR_TAR", os.path.join(REF, "vtkfiles.tar"))
sys.path.insert(0, SOLVER)  # report scripts import the solver modules
