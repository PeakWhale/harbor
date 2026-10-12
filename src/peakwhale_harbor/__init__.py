"""PeakWhale Harbor: turn a spreadsheet (CSV) into a prediction API."""

__version__ = "2.0.0"

from .bundle import Bundle, BundleError, load_bundle  # noqa: E402
from .schema import InputError  # noqa: E402
from .train import TrainingError, train  # noqa: E402

__all__ = [
    "Bundle",
    "BundleError",
    "InputError",
    "TrainingError",
    "__version__",
    "load_bundle",
    "train",
]
