"""Configuration and canonical phased CNV state space for `numbat`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

COLLAPSED_STATES: tuple[str, ...] = ("neu", "amp", "del", "loh", "bamp", "bdel")
SettingsType = Literal["default", "R_default"]


def validate_settings(settings: str) -> SettingsType:
    """Validate `settings` mode strictly ('default' or 'R_default')."""
    if settings not in ("default", "R_default"):
        raise ValueError(
            f"`settings` must be 'default' or 'R_default', got {settings!r}."
        )
    return settings  # type: ignore[return-value]


@dataclass(frozen=True)
class CNVStateSpec:
    """Specification of a single phased hidden CNV state."""

    name: str
    collapsed: str
    subclass: str
    orientation: int
    mu: float
    phi: float

    def __post_init__(self) -> None:
        if self.collapsed not in COLLAPSED_STATES:
            raise ValueError(
                f"Invalid collapsed state {self.collapsed!r}; expected one of {COLLAPSED_STATES}."
            )
        if self.orientation not in (-1, 0, 1):
            raise ValueError(
                f"Invalid orientation {self.orientation!r}; expected -1, 0, or 1."
            )
        if not (0.0 < self.mu < 1.0):
            raise ValueError(f"Expected allele fraction mu in (0, 1), got {self.mu}.")
        if self.phi <= 0.0:
            raise ValueError(f"Expected expression fold-change phi > 0, got {self.phi}.")


DEFAULT_STATES: tuple[CNVStateSpec, ...] = (
    CNVStateSpec(name="neu", collapsed="neu", subclass="neu", orientation=0, mu=0.50, phi=1.00),
    CNVStateSpec(name="del_p", collapsed="del", subclass="del", orientation=1, mu=0.90, phi=0.50),
    CNVStateSpec(name="del_m", collapsed="del", subclass="del", orientation=-1, mu=0.10, phi=0.50),
    CNVStateSpec(
        name="amp_p", collapsed="amp", subclass="amp", orientation=1, mu=2.0 / 3.0, phi=1.50
    ),
    CNVStateSpec(
        name="amp_m", collapsed="amp", subclass="amp", orientation=-1, mu=1.0 / 3.0, phi=1.50
    ),
    CNVStateSpec(name="amp2_p", collapsed="amp", subclass="amp2", orientation=1, mu=0.75, phi=2.00),
    CNVStateSpec(name="amp2_m", collapsed="amp", subclass="amp2", orientation=-1, mu=0.25, phi=2.00),
    CNVStateSpec(name="loh_p", collapsed="loh", subclass="loh", orientation=1, mu=0.90, phi=1.00),
    CNVStateSpec(name="loh_m", collapsed="loh", subclass="loh", orientation=-1, mu=0.10, phi=1.00),
    CNVStateSpec(name="bamp", collapsed="bamp", subclass="bamp", orientation=0, mu=0.50, phi=2.00),
    CNVStateSpec(name="bdel", collapsed="bdel", subclass="bdel", orientation=0, mu=0.50, phi=0.15),
)

# Exact 15-state catalog matching kharchenkolab/numbat v1.5.2 (hahmmr::run_joint_hmm_s15):
# 3 balanced unphased states (neu, bamp, bdel [00, phi < 1]) + 6 mirrored phased pairs
# covering clonal/subclonal mono-allelic deletions, clonal/subclonal CN-neutral LOH,
# and single/multi-level unbalanced amplifications.
S15_STATES: tuple[CNVStateSpec, ...] = (
    CNVStateSpec(name="neu", collapsed="neu", subclass="neu", orientation=0, mu=0.50, phi=1.00),
    CNVStateSpec(name="bamp", collapsed="bamp", subclass="bamp", orientation=0, mu=0.50, phi=2.00),
    CNVStateSpec(name="bdel", collapsed="bdel", subclass="bdel", orientation=0, mu=0.50, phi=0.20),
    CNVStateSpec(name="del_1_p", collapsed="del", subclass="del_1", orientation=1, mu=0.90, phi=0.50),
    CNVStateSpec(name="del_1_m", collapsed="del", subclass="del_1", orientation=-1, mu=0.10, phi=0.50),
    CNVStateSpec(
        name="del_2_p", collapsed="del", subclass="del_2", orientation=1, mu=2.0 / 3.0, phi=0.75
    ),
    CNVStateSpec(
        name="del_2_m", collapsed="del", subclass="del_2", orientation=-1, mu=1.0 / 3.0, phi=0.75
    ),
    CNVStateSpec(name="loh_1_p", collapsed="loh", subclass="loh_1", orientation=1, mu=0.90, phi=1.00),
    CNVStateSpec(name="loh_1_m", collapsed="loh", subclass="loh_1", orientation=-1, mu=0.10, phi=1.00),
    CNVStateSpec(name="loh_2_p", collapsed="loh", subclass="loh_2", orientation=1, mu=0.75, phi=1.00),
    CNVStateSpec(name="loh_2_m", collapsed="loh", subclass="loh_2", orientation=-1, mu=0.25, phi=1.00),
    CNVStateSpec(
        name="amp_1_p", collapsed="amp", subclass="amp_1", orientation=1, mu=2.0 / 3.0, phi=1.50
    ),
    CNVStateSpec(
        name="amp_1_m", collapsed="amp", subclass="amp_1", orientation=-1, mu=1.0 / 3.0, phi=1.50
    ),
    CNVStateSpec(name="amp_2_p", collapsed="amp", subclass="amp_2", orientation=1, mu=0.75, phi=2.25),
    CNVStateSpec(name="amp_2_m", collapsed="amp", subclass="amp_2", orientation=-1, mu=0.25, phi=2.25),
)


@dataclass(frozen=True)
class NumbatConfig:
    """Hyperparameters controlling joint HMM inference, segmentation, and phylogeny."""

    t_cnv: float = 1e-5
    gamma: float = 20.0
    nu: float = 1.0
    p_init_cnv: float = 0.15
    expr_model: Literal["pln", "gaussian_logratio"] = "pln"
    n_quadrature: int = 21
    w_expr: float = 1.0
    w_allele: float = 1.0
    min_loci: int = 5
    min_seg_prob: float = 0.5
    max_entropy: float = 0.5
    min_log_bf: float = 1.0
    seg_prior: float = 0.5
    tumor_prob_threshold: float = 0.5
    tau: float = 0.3
    min_cells: int | None = None
    max_tree_cells: int = 400
    max_nni_iter: int = 50
    max_iter: int = 2
    sigma_min: float = 0.15
    sigma_max: float = 2.5
    default_sigma: float = 0.4
    backend: Literal["auto", "numba", "numpy"] = "auto"
    states: tuple[CNVStateSpec, ...] | None = None
    settings: SettingsType = "default"

    def __post_init__(self) -> None:
        validate_settings(self.settings)
        if self.states is None:
            object.__setattr__(
                self,
                "states",
                S15_STATES if self.settings == "R_default" else DEFAULT_STATES,
            )
        else:
            object.__setattr__(self, "states", tuple(self.states))
        if not (0.0 < self.t_cnv < 0.5):
            raise ValueError(f"t_cnv must be in (0, 0.5), got {self.t_cnv}.")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}.")
        if self.nu < 0.0:
            raise ValueError(f"nu must be >= 0, got {self.nu}.")
        if not (0.0 < self.p_init_cnv < 1.0):
            raise ValueError(f"p_init_cnv must be in (0, 1), got {self.p_init_cnv}.")
        if self.expr_model not in ("pln", "gaussian_logratio"):
            raise ValueError(f"Unsupported expr_model {self.expr_model!r}.")
        if self.n_quadrature < 3:
            raise ValueError(f"n_quadrature must be >= 3, got {self.n_quadrature}.")
        if self.w_expr < 0.0 or self.w_allele < 0.0:
            raise ValueError("w_expr and w_allele must be non-negative.")
        if self.min_loci < 1:
            raise ValueError(f"min_loci must be >= 1, got {self.min_loci}.")
        if not (0.0 <= self.min_seg_prob <= 1.0):
            raise ValueError(f"min_seg_prob must be in [0, 1], got {self.min_seg_prob}.")
        if self.max_entropy <= 0.0:
            raise ValueError(f"max_entropy must be > 0, got {self.max_entropy}.")
        if self.min_log_bf < 0.0:
            raise ValueError(f"min_log_bf must be >= 0, got {self.min_log_bf}.")
        if not (0.0 < self.seg_prior < 1.0):
            raise ValueError(f"seg_prior must be in (0, 1), got {self.seg_prior}.")
        if not (0.0 < self.tumor_prob_threshold < 1.0):
            raise ValueError(
                f"tumor_prob_threshold must be in (0, 1), got {self.tumor_prob_threshold}."
            )
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}.")
        if self.min_cells is not None and self.min_cells < 1:
            raise ValueError(f"min_cells must be >= 1 when specified, got {self.min_cells}.")
        if self.max_tree_cells < 3:
            raise ValueError(f"max_tree_cells must be >= 3, got {self.max_tree_cells}.")
        if self.max_nni_iter < 0:
            raise ValueError(f"max_nni_iter must be >= 0, got {self.max_nni_iter}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.sigma_min <= 0.0 or self.sigma_max < self.sigma_min:
            raise ValueError(
                f"Invalid sigma bounds [{self.sigma_min}, {self.sigma_max}]."
            )
        if not (self.sigma_min <= self.default_sigma <= self.sigma_max):
            raise ValueError(
                f"default_sigma ({self.default_sigma}) must lie in "
                f"[{self.sigma_min}, {self.sigma_max}]."
            )
        if self.backend not in ("auto", "numba", "numpy"):
            raise ValueError(f"backend must be 'auto', 'numba', or 'numpy', got {self.backend!r}.")
        if len(self.states) < 2:
            raise ValueError("states must contain at least 2 states.")
        names = [s.name for s in self.states]
        if len(set(names)) != len(names):
            raise ValueError("State names in config.states must be unique.")
        if "neu" not in {s.collapsed for s in self.states}:
            raise ValueError("config.states must include a 'neu' collapsed state.")
