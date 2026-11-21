"""Pytest suite: physics bounds, retrieval ranking, the zero-hallucination
guarantee, spoof robustness, and figure export."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.alert_benchmark_gen import generate
from pipeline.phys_verifier import PhysicsVerifierGate
from pipeline.retriever import HybridRetriever
from pipeline.triage_engine import TriageEngine


def test_physics_bounds():
    v = PhysicsVerifierGate()
    assert v.verify({"accel_mps2": 10.0, "min_ttc_s": 9, "headway_s": 2, "phase_conflict": 0,
                     "spillback_rate": 0.1, "pos_residual_m": 0.3, "time_skew_s": 0.0})["regime"] \
        == "HIGH_KINEMATIC_RISK"
    benign = v.verify({"accel_mps2": 1.2, "min_ttc_s": 9, "headway_s": 2, "phase_conflict": 0,
                       "spillback_rate": 0.1, "pos_residual_m": 0.3, "time_skew_s": 0.0})
    assert benign["regime"] == "BENIGN_NOISE" and benign["proof"] == []
    sig = v.verify({"accel_mps2": 1, "min_ttc_s": 9, "headway_s": 2, "phase_conflict": 1,
                    "spillback_rate": 0.1, "pos_residual_m": 0.3, "time_skew_s": 0.0})
    assert sig["regime"] == "SIGNAL_CONFLICT"


def test_kb_loaded():
    r = HybridRetriever()
    assert len(r.kb) == 60
    assert all("rule" in s and "action_code" in s and "regime_tags" in s for s in r.kb)


def test_retrieval_regime_conditioning():
    r = HybridRetriever()
    q = "sudden hard braking deceleration emergency"
    plain = r.retrieve(q, regime_tags=None, k=3)
    boosted = r.retrieve(q, regime_tags=["HIGH_KINEMATIC_RISK"], k=3)
    # boosting a regime should surface SOPs of that regime
    assert any("HIGH_KINEMATIC_RISK" in r.by_id[s]["regime_tags"] for s in boosted)
    assert boosted != plain or all("HIG" in s for s in boosted)


def test_zero_hallucination_guarantee():
    inc = generate(n=120, seed=7)
    eng = TriageEngine()
    assert all(eng.physrag(x)["hallucinated"] == 0 for x in inc)   # by construction


def test_physrag_catches_spoofed_critical():
    inc = [x for x in generate(n=400, seed=7) if x["spoofed"] and x["is_critical"]]
    assert inc, "need spoofed critical incidents"
    eng = TriageEngine()
    phys_hits = sum(eng.physrag(x)["severity_tier"] == "CRITICAL" for x in inc)
    std_hits = sum(eng.standard_rag(x)["severity_tier"] == "CRITICAL" for x in inc)
    assert phys_hits > std_hits                                   # physics beats telemetry-trust
    assert phys_hits / len(inc) > 0.8


def test_physrag_severity_is_physics_grounded():
    inc = generate(n=80, seed=3)
    eng = TriageEngine()
    v = PhysicsVerifierGate()
    for x in inc:
        tk = eng.physrag(x)
        assert tk["severity_tier"] == v.verify(x["phys_observed"])["severity_floor"]


def test_generate_fields():
    inc = generate(n=50, seed=1)
    assert len(inc) == 50
    for x in inc:
        assert set(["phys", "reported", "gt_sop_ids", "true_severity", "query_text"]) <= set(x)
        assert 0.0 <= x["phys"]["spillback_rate"] <= 1.0


def test_figures_from_fixtures(tmp_path, monkeypatch):
    import pandas as pd
    import visualization.make_pub_figures as mpf
    res = tmp_path / "results"; fig = tmp_path / "figures"; res.mkdir(); fig.mkdir()
    methods = ["Rule-Only", "Standard RAG", "PhysRAG-V2X"]
    pd.DataFrame([{"method": m, "retrieval_recall_at_3": 0.6, "severity_macro_f1": 0.9,
                   "critical_fn_rate": 0.4 if m != "PhysRAG-V2X" else 0.02,
                   "hallucinated_action_pct": 16 if m == "Standard RAG" else 0,
                   "latency_ms": 1.5, "critical_fn_rate_spoofed": 1.0 if m != "PhysRAG-V2X" else 0.03}
                  for m in methods]).to_csv(res / "triage_metrics.csv", index=False)
    rng = np.random.default_rng(0)
    sev = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
    rows = []
    for m in methods:
        for _ in range(120):
            rows.append({"incident_id": f"INC-{_:04d}", "method": m, "spoofed": 0, "is_critical": 1,
                         "true_severity": rng.choice(sev), "pred_severity": rng.choice(sev),
                         "quarantine_action": "X", "sop_citation_id": "SOP", "hallucinated": 0,
                         "recall_at_3": 0.6, "latency_ms": 1.5})
    pd.DataFrame(rows).to_csv(res / "tickets.csv", index=False)
    pd.DataFrame([{"component": c, "latency_ms": v} for c, v in
                  [("Physics Gate", 0.01), ("Retrieval", 1.4), ("Ticket Synthesis", 0.05)]]
                 ).to_csv(res / "latency_breakdown.csv", index=False)
    pd.DataFrame([{"alert_volume": v, "method": m, "false_escalations": int(rng.integers(0, 20))}
                  for v in range(40, 401, 40) for m in methods]).to_csv(
        res / "workload_curve.csv", index=False)
    monkeypatch.setattr(mpf, "RESULTS", str(res)); monkeypatch.setattr(mpf, "FIGDIR", str(fig))
    mpf.main()
    for name in ("fig1_hallucination_vs_recall", "fig2_regime_confusion_matrix",
                 "fig3_latency_breakdown", "fig4_operator_workload_reduction"):
        for ext in ("svg", "pdf", "jpg"):
            assert (fig / f"{name}.{ext}").exists()
