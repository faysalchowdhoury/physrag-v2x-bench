# Physics-Verified Retrieval-Augmented Triage for Connected-Vehicle Cyber-Physical Incident Response (PhysRAG-V2X)

A reproducible, fully offline benchmark for triaging V2X security alerts at a
Traffic Management Center (TMC). When standard Retrieval-Augmented Generation
(RAG) over SAE J2735/J2945-1 standards and incident playbooks trusts reported
telemetry, it hallucinates false severities and **misses spoofed critical
events**. **PhysRAG-V2X** gates retrieval behind a deterministic kinematic/LWR
**physics verifier**, conditions retrieval on the verified physical regime, and
synthesizes **constrained** JSON triage tickets whose actions must be backed by a
SOP whose rule actually fires on the verified physics — a zero-hallucination
guarantee, robust to telemetry spoofing.

> **Scope & honesty.** No public dataset of V2X incident-triage tickets exists,
> so this is a **constructed benchmark**: physics-based synthetic telemetry (with
> spoofed cases) over a curated 60-entry SOP knowledge base citing real SAE/DOT
> clauses. The response synthesizer is a deterministic offline adapter — **no
> API keys, no network, fully reproducible**. The verifier is evaluated given
> physical quantities recovered with finite precision; field performance depends
> on that recovery (e.g. kinematic-consistency estimators) and will sit below the
> ceiling reported here. Results include the honest cases (borderline errors, a
> near- but not-perfect severity F1) rather than a clean sweep.

## Framework

1. **Deterministic Kinematic/LWR Verifier Gate** (`pipeline/phys_verifier.py`) —
   exact physical violation bounds: acceleration limit, headway feasibility,
   time-to-collision, signal-phase conflict, queue-spillback rate, GNSS residual,
   timestamp skew → a physical **regime** (`HIGH_KINEMATIC_RISK`, `SIGNAL_CONFLICT`,
   `SENSOR_DRIFT`, `BENIGN_NOISE`) and a machine-checkable `physical_proof`.
2. **Physics-Conditioned Hybrid Retriever** (`pipeline/retriever.py`) — BM25 +
   dense TF-IDF cosine over 60 SOPs, with the verified regime boosting matching
   SOPs so a spoofed, benign-looking alert is still routed to the correct
   high-severity playbook.
3. **Constrained Structured Synthesizer** (`pipeline/triage_engine.py`) — emits a
   JSON ticket (`severity_tier`, `quarantine_action`, `sop_citation_id`,
   `physical_proof`); the action may only come from a retrieved SOP whose rule
   fires on the verified physics (else an explicit `FLAG_FOR_REVIEW`).

**Baselines:** `Rule-Only` threshold triage on reported telemetry;
`Standard RAG` (unconstrained BM25/dense over the reported alert text).

## Benchmark (400 incidents: ghost-braking, phantom-vehicle, SPaT-spoof, sensor-drift, benign-noise)

| Method | Retrieval Recall@3 | Severity F1 | Critical FN | Hallucinated Action % | Latency (ms) | Critical FN (spoofed) |
|---|---:|---:|---:|---:|---:|---:|
| Rule-Only | — | 0.773 | 0.429 | 0.0 | 0.01 | **1.00** |
| Standard RAG | 0.594 | 0.773 | 0.429 | 16.5 | 1.49 | **1.00** |
| **PhysRAG-V2X** | **0.701** | **0.988** | **0.019** | **0.0** | 1.63 | **0.03** |

### What holds, and what does not (honest)

- **The physics gate is the whole point under spoofing.** On spoofed incidents
  (reported telemetry masks the hazard), Rule-Only and Standard RAG miss **100%**
  of critical events (`critical FN spoofed = 1.00`); PhysRAG misses **3%**
  (`0.03`), because it classifies severity from verified physics, not the spoofed
  report (`fig2` — Standard RAG dumps 43% of CRITICAL and 44% of HIGH incidents
  into LOW).
- **Zero hallucinated actions by construction.** PhysRAG only emits an action
  cited to a SOP whose rule fires on verified physics (0.0%); Standard RAG asserts
  physically-unsupported actions 16.5% of the time.
- **Not a clean sweep.** PhysRAG severity F1 is **0.988, not 1.0** — borderline
  incidents near the physical thresholds, observed with finite precision, are
  occasionally misclassified (critical FN 0.019). This is expected and reported.
- **On non-spoofed alerts the baselines are fine.** Standard RAG matches
  Rule-Only severity F1 (0.773); the gap is created specifically by spoofing and
  by consequence-blind severity assignment.
- **Latency.** The physics gate adds negligible cost; retrieval dominates
  (`fig3`). All methods triage in ~1-2 ms.
- **Operator workload.** Under alert-volume surges PhysRAG raises far fewer false
  escalations of benign/sensor-drift incidents (`fig4`).

## Figures (`figures/`, 300 DPI, svg/pdf/jpg)

1. `fig1_hallucination_vs_recall` — critical recall vs. hallucinated-action rate.
2. `fig2_regime_confusion_matrix` — severity confusion under spoofed telemetry
   (Standard RAG vs. PhysRAG).
3. `fig3_latency_breakdown` — per-stage latency (gate / retrieval / synthesis).
4. `fig4_operator_workload_reduction` — false escalations vs. alert volume.

## Install & reproduce

```bash
pip install -r requirements.txt
python kb/build_corpus.py                 # (re)build kb/v2x_sop_corpus.json (60 SOPs)
pytest -q                                 # 8 tests: physics, retrieval, zero-hallucination, export
python experiments/eval_triage.py         # 400 incidents -> results/*.csv (+ console table)
python visualization/make_pub_figures.py  # figures/ at 300 DPI (svg/pdf/jpg)
```

Fully offline and deterministic (fixed seeds); no external API keys.

## Repository structure

```
kb/build_corpus.py              generator for the 60-entry SOP knowledge base
kb/v2x_sop_corpus.json          structured SOPs (regime, rule, action, severity, citation)
data_gen/alert_benchmark_gen.py 400 incidents across 5 categories, reported vs. physical telemetry
pipeline/phys_verifier.py       deterministic kinematic/LWR verifier gate
pipeline/retriever.py           physics-conditioned BM25 + dense TF-IDF retriever
pipeline/triage_engine.py       rule-only, standard RAG, and PhysRAG-V2X + offline synthesizer
experiments/eval_triage.py      consequence-aware triage metrics
visualization/make_pub_figures.py four 300-DPI figures
tests/test_physrag.py           physics, retrieval, zero-hallucination, spoof robustness, export
```

## Limitations & threats to validity

- **Constructed benchmark.** Telemetry and the benign/spoofed mix are synthetic;
  absolute rates are assumptions. The relative comparison under a shared generator
  is the contribution, not field measurement.
- **Physics-recoverability assumption.** The gate is given physical quantities
  observed with finite precision; in the field these must be recovered (e.g. from
  kinematic consistency), which bounds real performance below this ceiling.
- **Deterministic synthesizer.** The offline adapter is rule-grounded; an actual
  LLM backend would add variance (and is where the zero-hallucination constraint
  matters most).

## License

MIT — see [LICENSE](LICENSE).
