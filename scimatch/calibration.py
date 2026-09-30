"""Disjoint Dcal Wilson map, with monotonic upper values across base-score bins."""
from dataclasses import dataclass
import numpy as np


def wilson_upper(positives: int, n: int, z: float = 1.645) -> float:
    if n < 1 or positives < 0 or positives > n:
        raise ValueError("invalid bin counts")
    p = positives / n
    return float((p + z*z/(2*n) + z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


@dataclass
class WilsonMap:
    edges: np.ndarray
    upper: np.ndarray
    z: float = 1.645

    @classmethod
    def fit(cls, probabilities, labels, bins: int = 10, z: float = 1.645):
        p, y = np.asarray(probabilities, dtype=float), np.asarray(labels, dtype=int)
        if p.ndim != 1 or y.shape != p.shape or len(p) < bins:
            raise ValueError("Dcal needs at least one observation per bin")
        if np.any((p < 0) | (p > 1)) or not np.isin(y, [0, 1]).all():
            raise ValueError("invalid probabilities or labels")
        order = np.argsort(p, kind="stable")
        chunks = np.array_split(order, bins)
        # Equal-frequency grouping by sorted rank, then threshold lookup at inference.
        edges = np.array([np.max(p[ix]) for ix in chunks[:-1]], dtype=float)
        upper = np.maximum.accumulate([wilson_upper(int(y[ix].sum()), len(ix), z) for ix in chunks])
        return cls(edges, np.asarray(upper), z)

    def transform(self, probabilities):
        p = np.asarray(probabilities, dtype=float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities outside [0,1]")
        return self.upper[np.searchsorted(self.edges, p, side="left")]

    def state_dict(self):
        return {"edges": self.edges.tolist(), "upper": self.upper.tolist(), "z": self.z}

    @classmethod
    def from_dict(cls, data):
        return cls(np.asarray(data["edges"], dtype=float), np.asarray(data["upper"], dtype=float), data["z"])
