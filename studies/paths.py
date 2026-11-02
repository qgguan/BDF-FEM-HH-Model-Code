"""Project-relative defaults shared by independently runnable studies."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDIES_ROOT = PROJECT_ROOT / "studies"
MANUFACTURED_RESULTS = STUDIES_ROOT / "manufactured" / "results"
PROPAGATION_RESULTS = STUDIES_ROOT / "propagation" / "results"
REFINEMENT_RESULTS = STUDIES_ROOT / "refinement" / "results"
WAVE_SPEED_RESULTS = STUDIES_ROOT / "wave_speed" / "results"
QUADRATURE_RESULTS = STUDIES_ROOT / "quadrature" / "results"
HH_CONVERGENCE_RESULTS = STUDIES_ROOT / "hh_convergence" / "results"


def relative_data_path(path, directory):
    """Write portable data-store references relative to their manifest."""
    return Path(os.path.relpath(path, directory)).as_posix()
