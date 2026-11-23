"""Where the report scripts find the solver, its results and the data.

Defaults follow the repository layout; override them with environment variables when the
snapshots live elsewhere (e.g. on scratch storage):

    QCFD_SOLVER         directory with chorin_projection.py and bundle_simulation.py
    QCFD_DATA           the data directory
    QCFD_RESULTS        a run's <tag>_params.pfile and temp/ snapshots
    QCFD_REFERENCE      Star-CCM+ C_D.csv, C_L.csv and star_t*.vtk
    QCFD_REPORT_DATA    extractor outputs, read by plots.py
    QCFD_STAR_TAR       full Star-CCM+ snapshot archive (not shipped); only needed for
                        snapshot times that are not cached as data/star_ccm/star_t*.vtk
    IC                  warm-start file, overrides the standard name
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVER = os.environ.get("QCFD_SOLVER", os.path.join(ROOT, "scripts"))
DATA = os.environ.get("QCFD_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("QCFD_RESULTS", os.path.join(DATA, "results"))
IC_DIR = os.path.join(DATA, "initial_conditions")
MASK_DIR = os.path.join(DATA, "mps_masks")
REF = os.environ.get("QCFD_REFERENCE", os.path.join(DATA, "star_ccm"))
OUT = os.environ.get("QCFD_REPORT_DATA", os.path.join(DATA, "archive"))
STAR_TAR = os.environ.get("QCFD_STAR_TAR", os.path.join(REF, "vtkfiles.tar"))

sys.path.insert(0, SOLVER)  # report scripts import the solver modules


def warm_start_file(p, Lf):
    return os.path.expanduser(os.environ.get("IC", f"{IC_DIR}/enec_ic_p{p}_f{Lf}_d500.mps"))


def mask_file(p, Lf):
    return f"{MASK_DIR}/enec_mask_p{p}_f{Lf}_d500.mps"
