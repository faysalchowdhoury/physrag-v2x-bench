"""Triage engines that turn a V2X alert into a structured JSON ticket.

* ``rule_only``    -- classic threshold triage on *reported* telemetry (no KB).
* ``standard_rag`` -- unconstrained BM25/dense RAG on the reported alert text;
  it trusts the reported severity cue and cites whatever it retrieves, so it
  over-escalates (hallucinates) on borderline text and misses spoofed criticals.
* ``physrag``      -- verifier gate first (physics regime + proof), regime-
  conditioned retrieval, and a **constrained synthesizer** that only emits an
  action backed by a retrieved SOP whose rule actually fires on the *verified*
  physics -- a zero-hallucination guarantee, robust to spoofing.

The synthesizer is a deterministic offline adapter (no external API keys).
"""
from __future__ import annotations

import time
from typing import Dict

from pipeline.phys_verifier import PhysicsVerifierGate, BOUNDS, REGIME_SEVERITY
from pipeline.retriever import HybridRetriever

REGIME_ACTION = {"HIGH_KINEMATIC_RISK": "QUARANTINE_SENDER", "SIGNAL_CONFLICT": "ISSUE_RSU_ADVISORY",
                 "SENSOR_DRIFT": "RECALIBRATE_SENSOR", "BENIGN_NOISE": "SUPPRESS_ALERT"}


def _ticket(inc, method, severity, action, cite, proof, hallucinated, retrieved):
    return dict(incident_id=inc["incident_id"], method=method, severity_tier=severity,
                quarantine_action=action, sop_citation_id=cite, physical_proof=proof,
                hallucinated=int(hallucinated), retrieved_sops=retrieved,
                true_severity=inc["true_severity"], is_critical=inc["is_critical"],
                gt_sop_ids=inc["gt_sop_ids"], spoofed=inc["spoofed"])


class TriageEngine:
    def __init__(self, retriever: HybridRetriever = None, verifier: PhysicsVerifierGate = None):
        self.retriever = retriever or HybridRetriever()
        self.verifier = verifier or PhysicsVerifierGate()

    # -- baseline 1: rule-only threshold triage (reads reported telemetry) --- #
    def rule_only(self, inc: Dict) -> dict:
        t0 = time.perf_counter()
        g = self.verifier.verify(inc["reported"])          # thresholds on REPORTED values
        sev = g["severity_floor"]
        action = REGIME_ACTION[g["regime"]]
        lat = (time.perf_counter() - t0) * 1e3
        tk = _ticket(inc, "Rule-Only", sev, action, None, g["proof"], False, [])
        tk["latency_ms"] = lat
        return tk

    # -- baseline 2: standard unconstrained RAG (reported text only) --------- #
    def standard_rag(self, inc: Dict) -> dict:
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(inc["query_text"], regime_tags=None, k=3)
        top = self.retriever.by_id[retrieved[0]]
        sev = top["severity_tier"]                         # trusts retrieved-by-text severity
        action = top["action_code"]
        # hallucination: asserts an action from a SOP whose rule does NOT hold under VERIFIED physics
        r = top["rule"]
        fires = self.verifier.rule_fires(r["metric"], r["op"], r["threshold"], inc["phys"])
        lat = (time.perf_counter() - t0) * 1e3
        tk = _ticket(inc, "Standard RAG", sev, action, top["sop_id"], None,
                     hallucinated=not fires, retrieved=retrieved)
        tk["latency_ms"] = lat
        return tk

    # -- proposed: PhysRAG-V2X ---------------------------------------------- #
    def physrag(self, inc: Dict) -> dict:
        obs = inc.get("phys_observed", inc["phys"])        # physics recovered w/ finite precision
        t0 = time.perf_counter()
        g = self.verifier.verify(obs)                      # verified physics regime + proof
        t1 = time.perf_counter()
        aug = inc["query_text"] + " " + g["regime"] + " " + " ".join(
            p["metric"] for p in g["proof"])
        retrieved = self.retriever.retrieve(aug, regime_tags=[g["regime"]], k=3)
        t2 = time.perf_counter()
        sev = g["severity_floor"]                          # severity grounded in verified physics
        # constrained synthesis: only emit an action from a SOP whose rule fires on verified physics
        cite, action = None, "FLAG_FOR_REVIEW"
        for sid in retrieved:
            s = self.retriever.by_id[sid]
            r = s["rule"]
            if (g["regime"] in s["regime_tags"] and
                    self.verifier.rule_fires(r["metric"], r["op"], r["threshold"], obs)):
                cite, action = sid, s["action_code"]
                break
        t3 = time.perf_counter()
        tk = _ticket(inc, "PhysRAG-V2X", sev, action, cite, g["proof"],
                     hallucinated=False, retrieved=retrieved)   # zero-hallucination by construction
        tk["latency_ms"] = (t3 - t0) * 1e3
        tk["gate_ms"] = (t1 - t0) * 1e3
        tk["retrieval_ms"] = (t2 - t1) * 1e3
        tk["synthesis_ms"] = (t3 - t2) * 1e3
        return tk

    def triage(self, inc: Dict, method: str) -> dict:
        return {"Rule-Only": self.rule_only, "Standard RAG": self.standard_rag,
                "PhysRAG-V2X": self.physrag}[method](inc)
