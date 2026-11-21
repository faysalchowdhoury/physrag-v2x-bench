"""Generates realistic V2X telemetry incident scenarios for triage benchmarking.

Each incident carries both **reported** telemetry (what an unconstrained RAG
reads, and which an attacker can spoof) and the **physical** ground-truth
quantities a kinematic/LWR verifier recovers. In *spoofed* incidents the reported
fields look benign while the physics is dangerous -- the case that makes
telemetry-trusting triage hallucinate a low severity and miss a critical event.

Five categories: ghost-braking, phantom-vehicle, SPaT-spoof, sensor-drift, and
benign-noise edge cases. Ground-truth severity, physical regime, and the set of
applicable SOPs are attached to every incident.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

CATEGORIES = ["Ghost_Braking", "Phantom_Vehicle", "SPaT_Spoof", "Sensor_Drift", "Benign_Noise"]
CAT_REGIME = {
    "Ghost_Braking": "HIGH_KINEMATIC_RISK", "Phantom_Vehicle": "HIGH_KINEMATIC_RISK",
    "SPaT_Spoof": "SIGNAL_CONFLICT", "Sensor_Drift": "SENSOR_DRIFT",
    "Benign_Noise": "BENIGN_NOISE",
}
REGIME_SEVERITY = {"HIGH_KINEMATIC_RISK": "CRITICAL", "SIGNAL_CONFLICT": "HIGH",
                   "SENSOR_DRIFT": "MODERATE", "BENIGN_NOISE": "LOW"}
CRITICAL_REGIMES = {"HIGH_KINEMATIC_RISK"}

_KB = None


def _load_kb():
    global _KB
    if _KB is None:
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "kb", "v2x_sop_corpus.json")
        with open(p, "r", encoding="utf-8") as fh:
            _KB = json.load(fh)
    return _KB


def _applicable_sops(regime: str, phys: Dict[str, float]) -> List[str]:
    hits = []
    for s in _load_kb():
        if regime not in s["regime_tags"]:
            continue
        r = s["rule"]
        v = phys.get(r["metric"])
        if v is None:
            continue
        ok = (v > r["threshold"]) if r["op"] == ">" else (v < r["threshold"])
        if ok:
            hits.append(s["sop_id"])
    return hits


def _phys_for(cat: str, rng, borderline: bool = False) -> Dict[str, float]:
    """True physical quantities per category (``borderline`` straddles the bound)."""
    p = dict(accel_mps2=abs(rng.normal(1.5, 0.6)), headway_s=rng.uniform(1.2, 2.5),
             min_ttc_s=rng.uniform(4.0, 9.0), speed_mps=rng.uniform(18, 33),
             phase_conflict=0.0, spillback_rate=rng.uniform(0.05, 0.35),
             pos_residual_m=abs(rng.normal(0.4, 0.2)), time_skew_s=abs(rng.normal(0.05, 0.03)))
    if cat == "Ghost_Braking":
        p["accel_mps2"] = rng.uniform(7.3, 8.7) if borderline else rng.uniform(8.8, 11.5)
        p["headway_s"] = rng.uniform(0.3, 0.6)
    elif cat == "Phantom_Vehicle":
        p["min_ttc_s"] = rng.uniform(1.3, 1.7) if borderline else rng.uniform(0.5, 1.3)
        p["headway_s"] = rng.uniform(0.3, 0.7); p["accel_mps2"] = rng.uniform(5.0, 7.5)
    elif cat == "SPaT_Spoof":
        if borderline:
            p["phase_conflict"] = 0.0; p["spillback_rate"] = rng.uniform(0.72, 0.88)
        else:
            p["phase_conflict"] = 1.0; p["spillback_rate"] = rng.uniform(0.6, 0.95)
    elif cat == "Sensor_Drift":
        p["pos_residual_m"] = rng.uniform(2.2, 2.8) if borderline else rng.uniform(2.9, 5.0)
        p["time_skew_s"] = rng.uniform(0.25, 0.8)
    return p


def _reported(phys: Dict[str, float], spoofed: bool, rng) -> Dict[str, float]:
    """What the TMC/BSM reports: benign-looking under spoofing, else ~physical."""
    rep = dict(phys)
    if spoofed:
        rep["accel_mps2"] = abs(rng.normal(1.5, 0.5))      # attacker masks the hazard
        rep["headway_s"] = rng.uniform(1.4, 2.4)
        rep["min_ttc_s"] = rng.uniform(4.0, 8.0)
        rep["phase_conflict"] = 0.0
        rep["spillback_rate"] = rng.uniform(0.1, 0.35)
    else:
        for k in rep:
            rep[k] = rep[k] * (1.0 + rng.normal(0, 0.04))   # honest sensor noise
    return rep


_APPARENT_KW = {
    "HIGH_KINEMATIC_RISK": "sudden hard braking deceleration emergency kinematic hazard ghost braking",
    "SIGNAL_CONFLICT": "signal phase SPaT conflict intersection red light running spillback queue",
    "SENSOR_DRIFT": "GPS GNSS position drift residual sensor calibration localization",
    "BENIGN_NOISE": "nominal benign jitter noise within tolerance routine normal",
}


def _regime_of(v: Dict[str, float]) -> str:
    """Physical regime implied by a telemetry dict (same bounds as the verifier)."""
    if v["accel_mps2"] > 8.0 or v["min_ttc_s"] < 1.5 or v["headway_s"] < 0.5:
        return "HIGH_KINEMATIC_RISK"
    if v["phase_conflict"] > 0.5 or v["spillback_rate"] > 0.8:
        return "SIGNAL_CONFLICT"
    if v["pos_residual_m"] > 2.5 or v["time_skew_s"] > 0.3:
        return "SENSOR_DRIFT"
    return "BENIGN_NOISE"


def _query_text(cat: str, rep: Dict[str, float]) -> str:
    kw = _APPARENT_KW[_regime_of(rep)]               # phenomenon keywords from REPORTED signal
    return (f"V2X incident alert: {kw}. Telemetry — acceleration {rep['accel_mps2']:.1f} m/s2, "
            f"headway {rep['headway_s']:.2f} s, min TTC {rep['min_ttc_s']:.1f} s, "
            f"phase conflict {rep['phase_conflict']:.0f}, spillback {rep['spillback_rate']:.2f}, "
            f"position residual {rep['pos_residual_m']:.1f} m, timestamp skew "
            f"{rep['time_skew_s']:.2f} s.")


def generate(n: int = 400, seed: int = 20260101, spoof_frac: float = 0.4) -> List[dict]:
    _load_kb()
    rng = np.random.default_rng(seed)
    incidents = []
    for i in range(n):
        cat = CATEGORIES[i % len(CATEGORIES)]
        borderline = cat != "Benign_Noise" and rng.random() < 0.18
        phys = _phys_for(cat, rng, borderline)
        regime = _regime_of(phys)                           # TRUE regime from physics
        # only the dangerous categories can be spoofed to look benign
        spoofed = (cat in ("Ghost_Braking", "Phantom_Vehicle", "SPaT_Spoof")
                   and rng.random() < spoof_frac)
        rep = _reported(phys, spoofed, rng)
        gt_sops = _applicable_sops(regime, phys)
        # what the verifier *observes*: physics recovered with finite precision
        phys_obs = {k: (v * (1.0 + rng.normal(0, 0.05)) if k != "phase_conflict" else v)
                    for k, v in phys.items()}
        incidents.append(dict(
            incident_id=f"INC-{i:04d}", category=cat, spoofed=int(spoofed), borderline=int(borderline),
            true_regime=regime, true_severity=REGIME_SEVERITY[regime],
            is_critical=int(regime in CRITICAL_REGIMES),
            phys=phys, phys_observed=phys_obs, reported=rep, gt_sop_ids=gt_sops,
            query_text=_query_text(cat, rep)))
    return incidents


if __name__ == "__main__":
    inc = generate()
    from collections import Counter
    print("incidents:", len(inc))
    print("by category:", Counter(x["category"] for x in inc))
    print("spoofed:", sum(x["spoofed"] for x in inc))
    print("with >=1 applicable SOP:", sum(len(x["gt_sop_ids"]) > 0 for x in inc))
    ex = inc[0]
    print("example gt_sops:", ex["gt_sop_ids"], "severity:", ex["true_severity"])
