"""csaw: Peak-free sliding-window differential ChIP-seq / ATAC-seq / CUT&RUN analysis."""

from .filtering import FilterResult, filter_windows_global, filter_windows_local
from .normalization import norm_factors, norm_offsets
from .ql_glm import glm_ql_fit_test
from .regions import MergeResult, combine_tests, get_best_test, merge_windows
from .utils import SettingsMode, ave_log_cpm, benjamini_hochberg, scaled_average, validate_settings
from .windows import WindowData, window_counts

__version__ = "0.1.0"

__all__ = [
    "FilterResult",
    "MergeResult",
    "SettingsMode",
    "WindowData",
    "ave_log_cpm",
    "benjamini_hochberg",
    "combine_tests",
    "filter_windows_global",
    "filter_windows_local",
    "get_best_test",
    "glm_ql_fit_test",
    "merge_windows",
    "norm_factors",
    "norm_offsets",
    "scaled_average",
    "validate_settings",
    "window_counts",
]
