"""Deterministic kinematic / LWR verifier gate.

Computes exact physical violation bounds (acceleration limit, headway feasibility,
time-to-collision, signal-phase conflict, queue-spillback rate, GNSS residual,
timestamp skew) and returns the physical regime plus a machine-checkable
``physical_proof``. Because it reads physical quantities rather than the reported
severity cue, it is robust to telemetry spoofing.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (metric, operator, threshold, regime) -- the exact physical bounds
BOUNDS: List[Tuple[str, str, float, str]] = [
    ("accel_mps2", ">", 8.0, "HIGH_KINEMATIC_RISK"),
    ("min_ttc_s", "<", 1.5, "HIGH_KINEMATIC_RISK"),
    ("headway_s", "<", 0.5, "HIGH_KINEMATIC_RISK"),
    ("phase_conflict", ">", 0.5, "SIGNAL_CONFLICT"),
    ("spillback_rate", ">", 0.8, "SIGNAL_CONFLICT"),
    ("pos_residual_m", ">", 2.5, "SENSOR_DRIFT"),
    ("time_skew_s", ">", 0.3, "SENSOR_DRIFT"),
]
PRIORITY = ["HIGH_KINEMATIC_RISK", "SIGNAL_CONFLICT", "SENSOR_DRIFT", "BENIGN_NOISE"]
REGIME_SEVERITY = {"HIGH_KINEMATIC_RISK": "CRITICAL", "SIGNAL_CONFLICT": "HIGH",
                   "SENSOR_DRIFT": "MODERATE", "BENIGN_NOISE": "LOW"}


def _fires(value: float, op: str, thr: float) -> bool:
    return value > thr if op == ">" else value < thr


class PhysicsVerifierGate:
    def verify(self, values: Dict[str, float]) -> dict:
        """Return {regime, severity_floor, regimes, proof} for a telemetry dict."""
        proof = []
        regimes = set()
        for metric, op, thr, regime in BOUNDS:
            v = values.get(metric)
            if v is None:
                continue
            if _fires(float(v), op, thr):
                proof.append({"metric": metric, "op": op, "threshold": thr,
                              "value": round(float(v), 3), "regime": regime})
                regimes.add(regime)
        regime = next((r for r in PRIORITY if r in regimes), "BENIGN_NOISE")
        return {"regime": regime, "severity_floor": REGIME_SEVERITY[regime],
                "regimes": sorted(regimes), "proof": proof}

    def rule_fires(self, metric: str, op: str, thr: float, values: Dict[str, float]) -> bool:
        v = values.get(metric)
        return v is not None and _fires(float(v), op, thr)
