"""Evaluate triage quality across Rule-Only, Standard RAG, and PhysRAG-V2X.

Metrics: SOP Retrieval Recall@3, Severity-classification macro-F1, Critical
False-Negative Rate, Hallucinated Action Rate (%), and Triage Latency (ms).
Outputs results/triage_metrics.csv, results/tickets.csv, results/workload_curve.csv.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_gen.alert_benchmark_gen import generate
from pipeline.retriever import HybridRetriever
from pipeline.phys_verifier import PhysicsVerifierGate
from pipeline.triage_engine import TriageEngine

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
METHODS = ["Rule-Only", "Standard RAG", "PhysRAG-V2X"]
SEV_ORDER = ["CRITICAL", "HIGH", "MODERATE", "LOW"]


def recall_at_k(retrieved, gt, k=3):
    if not gt:
        return np.nan
    inter = len(set(retrieved[:k]) & set(gt))
    return inter / min(k, len(gt))


def main():
    print("[1/3] generating 400 incidents + running triage ...")
    incidents = generate(n=400, seed=20260101)
    engine = TriageEngine(HybridRetriever(), PhysicsVerifierGate())
    tickets = []
    for inc in incidents:
        for m in METHODS:
            tk = engine.triage(inc, m)
            tk["recall_at_3"] = recall_at_k(tk["retrieved_sops"], inc["gt_sop_ids"])
            tickets.append(tk)
    tdf = pd.DataFrame(tickets)
    tdf_out = tdf[["incident_id", "method", "spoofed", "is_critical", "true_severity",
                   "severity_tier", "quarantine_action", "sop_citation_id", "hallucinated",
                   "recall_at_3", "latency_ms"]].rename(columns={"severity_tier": "pred_severity"})
    tdf_out.to_csv(os.path.join(RESULTS, "tickets.csv"), index=False)

    print("[2/3] aggregating metrics ...")
    rows = []
    for m in METHODS:
        g = tdf[tdf.method == m]
        crit = g[g.is_critical == 1]
        crit_fn = float((crit.severity_tier != "CRITICAL").mean())
        row = {"method": m,
               "retrieval_recall_at_3": float(g.recall_at_3.mean(skipna=True))
               if g.recall_at_3.notna().any() else np.nan,
               "severity_macro_f1": float(f1_score(g.true_severity, g.severity_tier,
                                                    labels=SEV_ORDER, average="macro",
                                                    zero_division=0)),
               "critical_fn_rate": crit_fn,
               "hallucinated_action_pct": float(g.hallucinated.mean() * 100.0),
               "latency_ms": float(g.latency_ms.mean())}
        # spoofed-only critical FN (the stress case)
        sc = g[(g.is_critical == 1) & (g.spoofed == 1)]
        row["critical_fn_rate_spoofed"] = float((sc.severity_tier != "CRITICAL").mean()) \
            if len(sc) else np.nan
        rows.append(row)
    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(RESULTS, "triage_metrics.csv"), index=False)

    print("[3/3] operator-workload curve (false escalations vs alert volume) ...")
    # false escalation = benign/sensor-drift incident raised to HIGH/CRITICAL
    benign_ids = {inc["incident_id"] for inc in incidents
                  if inc["true_regime"] in ("BENIGN_NOISE", "SENSOR_DRIFT")}
    wrows = []
    for vol in range(40, 401, 40):
        sub = tdf[tdf.incident_id.isin({f"INC-{i:04d}" for i in range(vol)})]
        for m in METHODS:
            gm = sub[(sub.method == m) & (sub.incident_id.isin(benign_ids))]
            esc = int(((gm.severity_tier.isin(["CRITICAL", "HIGH"])) | (gm.hallucinated == 1)).sum())
            wrows.append({"alert_volume": vol, "method": m, "false_escalations": esc})
    pd.DataFrame(wrows).to_csv(os.path.join(RESULTS, "workload_curve.csv"), index=False)

    # latency breakdown (PhysRAG components)
    pr = tdf[tdf.method == "PhysRAG-V2X"]
    pd.DataFrame([{"component": "Physics Gate", "latency_ms": float(pr.gate_ms.mean())},
                  {"component": "Retrieval", "latency_ms": float(pr.retrieval_ms.mean())},
                  {"component": "Ticket Synthesis", "latency_ms": float(pr.synthesis_ms.mean())}]
                 ).to_csv(os.path.join(RESULTS, "latency_breakdown.csv"), index=False)

    pd.set_option("display.width", 160)
    print("\n================ TRIAGE BENCHMARK (400 incidents) ================")
    print(mdf.round(3).to_string(index=False))
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
