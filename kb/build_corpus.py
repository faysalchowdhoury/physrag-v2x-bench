"""Builds kb/v2x_sop_corpus.json: 60 structured V2X cybersecurity / TMC incident
-response SOPs. Each entry carries a physical-regime tag, a machine-checkable
rule threshold, an action code, a severity tier, a standards citation, and
retrieval text. Run: ``python kb/build_corpus.py``.
"""
from __future__ import annotations

import json
import os

REGIMES = ["HIGH_KINEMATIC_RISK", "SIGNAL_CONFLICT", "SENSOR_DRIFT", "BENIGN_NOISE"]
ACTIONS = {
    "HIGH_KINEMATIC_RISK": ["QUARANTINE_SENDER", "REVOKE_PSEUDONYM", "ESCALATE_TMC"],
    "SIGNAL_CONFLICT": ["ISSUE_RSU_ADVISORY", "ESCALATE_TMC", "QUARANTINE_SENDER"],
    "SENSOR_DRIFT": ["RECALIBRATE_SENSOR", "FLAG_FOR_REVIEW"],
    "BENIGN_NOISE": ["SUPPRESS_ALERT", "NO_ACTION"],
}
SEVERITY = {
    "HIGH_KINEMATIC_RISK": "CRITICAL",
    "SIGNAL_CONFLICT": "HIGH",
    "SENSOR_DRIFT": "MODERATE",
    "BENIGN_NOISE": "LOW",
}
# per-regime SOP archetypes (title stem, metric, operator, threshold, vocabulary, citation)
ARCHE = {
    "HIGH_KINEMATIC_RISK": [
        ("Implausible hard-braking BSM", "accel_mps2", ">", 8.0,
         "sudden deceleration ghost braking emergency electronic brake light kinematic implausibility",
         "SAE J2945/1 §6.3 (MBF)"),
        ("Infeasible acceleration jump", "accel_mps2", ">", 6.0,
         "acceleration spike jerk physically infeasible powertrain limit spoofed speed",
         "SAE J2735 §BSM part I"),
        ("Critical headway collapse", "headway_s", "<", 0.5,
         "unsafe following headway time gap tailgating closing speed rear-end shockwave",
         "SAE J2945/1 §6.4"),
        ("Phantom lead vehicle injection", "min_ttc_s", "<", 1.5,
         "sybil phantom vehicle fabricated lead emergency braking time to collision",
         "IEEE 1609.2 misbehavior"),
        ("Speed beyond kinematic bound", "speed_mps", ">", 42.0,
         "impossible speed beyond vehicle class limit falsified velocity",
         "SAE J2735 §DE_Speed"),
    ],
    "SIGNAL_CONFLICT": [
        ("Spoofed SPaT green on red approach", "phase_conflict", ">", 0.5,
         "SPaT signal phase timing spoof falsified green red light running intersection conflict",
         "SAE J2735 §SPaT / MAP"),
        ("Conflicting movement authorization", "phase_conflict", ">", 0.5,
         "conflicting movement permissive phase RSU advisory intersection collision",
         "NTCIP 1202 signal controller"),
        ("Queue spillback over capacity", "spillback_rate", ">", 0.8,
         "downstream queue spillback saturation LWR shockwave gridlock arterial",
         "FHWA TMC incident playbook"),
        ("Dilemma-zone exploitation", "phase_conflict", ">", 0.5,
         "dilemma zone yellow trap signal manipulation forced stop",
         "MUTCD signal timing"),
    ],
    "SENSOR_DRIFT": [
        ("GNSS position drift anomaly", "pos_residual_m", ">", 2.5,
         "GPS GNSS position drift dead reckoning residual multipath degraded fix",
         "SAE J2945/1 §6.2"),
        ("Clock / timestamp skew", "time_skew_s", ">", 0.3,
         "timestamp skew clock desync stale message inter-arrival jitter",
         "IEEE 1609.3 timing"),
        ("Low-confidence localization", "pos_residual_m", ">", 1.8,
         "position confidence degraded accuracy covariance sensor calibration",
         "SAE J2945/1 §6.2"),
        ("Intermittent BSM dropout", "time_skew_s", ">", 0.5,
         "dropout missing beacon channel congestion gap recalibrate",
         "SAE J2945/1 §5"),
    ],
    "BENIGN_NOISE": [
        ("Nominal GPS jitter", "pos_residual_m", "<", 1.0,
         "benign gps jitter noise within tolerance normal nominal no action",
         "SAE J2945/1 §6.2"),
        ("Routine deceleration", "accel_mps2", "<", 3.0,
         "normal braking comfortable deceleration routine traffic benign",
         "SAE J2735 §BSM"),
        ("Flash-crowd congestion (benign)", "spillback_rate", "<", 0.5,
         "recurring congestion benign flash crowd peak demand expected",
         "FHWA congestion mgmt"),
        ("Tolerable timing jitter", "time_skew_s", "<", 0.15,
         "minor jitter acceptable latency within spec suppress alert",
         "IEEE 1609.3"),
    ],
}


def build():
    sops = []
    sid = 1
    # cycle archetypes to reach 60 entries with varied thresholds / actions / phrasing
    per_regime = 15
    for regime in REGIMES:
        arche = ARCHE[regime]
        for k in range(per_regime):
            title, metric, op, thr, vocab, cite = arche[k % len(arche)]
            variant = k // len(arche)
            # perturb threshold slightly across variants to diversify rules
            t = round(thr * (1.0 + 0.06 * variant * (1 if op == ">" else -1)), 3)
            action = ACTIONS[regime][k % len(ACTIONS[regime])]
            sev = SEVERITY[regime]
            text = (f"{title}. Physical regime {regime}. When {metric} {op} {t}, the "
                    f"condition indicates {vocab}. Verify the kinematic/physical proof, then "
                    f"apply action {action.replace('_', ' ').lower()} and cite {cite}. "
                    f"Severity tier {sev}. Keywords: {vocab}.")
            sops.append({
                "sop_id": f"SOP-{regime[:3]}-{sid:03d}",
                "title": f"{title} (variant {variant+1})" if variant else title,
                "regime_tags": [regime],
                "keywords": vocab.split(),
                "rule": {"metric": metric, "op": op, "threshold": t},
                "action_code": action,
                "severity_tier": sev,
                "standard_ref": cite,
                "text": text,
            })
            sid += 1
    return sops


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2x_sop_corpus.json")
    sops = build()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(sops, fh, indent=2)
    print(f"wrote {len(sops)} SOPs to {out}")
    from collections import Counter
    print("by regime:", Counter(s["regime_tags"][0] for s in sops))
