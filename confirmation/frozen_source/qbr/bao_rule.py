"""Pure Bao Eqs. (11)-(12) arithmetic; no estimator or simulation.

Source: https://arxiv.org/html/2508.11291v1
Article DOI: 10.1109/ICCCWorkshops67136.2025.11147210

``bao_rule_transfer`` names an adapted decision rule, not a reproduction of
Bao's trained semantic router, cache management, topology, or full method.
All time inputs and outputs are seconds. Alpha has units inverse seconds.
"""

from dataclasses import dataclass
import math
from numbers import Real


def _number(name: str, value: Real, *, minimum: float | None = None,
            maximum: float | None = None, positive: bool = False) -> float:
    """Validate real scalars without coercing strings or accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not bool or text")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class FusionParameters:
    """theta: dimensionless [0,1]; alpha_per_second: finite nonnegative 1/s."""

    theta: float
    alpha_per_second: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta", _number("theta", self.theta, minimum=0, maximum=1))
        object.__setattr__(self, "alpha_per_second", _number(
            "alpha_per_second", self.alpha_per_second, minimum=0))


@dataclass(frozen=True)
class EstimatedLatencies:
    """Caller-supplied causal edge/cloud latency estimates, finite seconds >= 0.

    Both estimates must use the same information set and timing origin. This
    arithmetic container cannot verify their provenance. They may include CPU
    waits and wireless delivery for BOTH routes in the transferred topology.
    It performs no conversion from fees, quality scores, bytes, or token counts.
    """

    edge_seconds: float
    cloud_seconds: float

    def __post_init__(self) -> None:
        for name in ("edge_seconds", "cloud_seconds"):
            object.__setattr__(self, name, _number(name, getattr(self, name), minimum=0))


@dataclass(frozen=True)
class BaoRuleDecision:
    """cloud=1 iff the UNCLIPPED fused score strictly exceeds theta.

    cloud=0 denotes our edge route, including exact ties. Bao Eq. (12) uses
    the opposite integer convention: its d=0 is the remote large model.
    fused_score is dimensionless, not a calibrated probability.
    """

    cloud: int
    fused_score: float
    latency_gap_seconds: float
    label: str = "Bao-rule transfer"


def bao_rule_transfer(semantic_probability: float, latencies: EstimatedLatencies,
                      parameters: FusionParameters) -> BaoRuleDecision:
    """Transfer Eqs. (11)-(12) to already-estimated edge/cloud latencies.

    p is the supplied semantic probability in [0,1], not automatically a
    difference of model grades or an expected shortfall. This function does
    not define, train, recalibrate, or reinterpret that semantic input.

    cloud = int(p - alpha * (T_cloud - T_edge) > theta).
    No clipping, epsilon tie band, monetary term, or extra quality term.
    """
    if not isinstance(latencies, EstimatedLatencies):
        raise TypeError("latencies must be EstimatedLatencies")
    if not isinstance(parameters, FusionParameters):
        raise TypeError("parameters must be FusionParameters")
    p = _number("semantic_probability", semantic_probability, minimum=0, maximum=1)
    gap = latencies.cloud_seconds - latencies.edge_seconds
    fused = p - parameters.alpha_per_second * gap
    _number("computed fused_score", fused)
    return BaoRuleDecision(int(fused > parameters.theta), fused, gap)
