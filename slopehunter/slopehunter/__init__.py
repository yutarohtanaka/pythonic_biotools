"""SlopeHunter: Model-based clustering adjustment for collider / index-event bias (Mahmoud et al., 2022)."""
from .core import harmonise_effects, hunt, sh_adj, shclust

__all__ = ["harmonise_effects", "hunt", "sh_adj", "shclust"]
