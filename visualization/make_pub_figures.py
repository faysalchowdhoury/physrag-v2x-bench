"""Publication figures (dark theme, Times New Roman, 300 DPI, svg/pdf/jpg)."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

FIG_BG, AX_BG, FG, GRID = "#0f0f1a", "#1a1a2e", "#d6d6e0", "#33334d"
C_OURS, C_STD, C_RULE, C_EM = "#ffd166", "#ff6b6b", "#4cc9f0", "#06d6a0"
MCOLOR = {"PhysRAG-V2X": C_OURS, "Standard RAG": C_STD, "Rule-Only": C_RULE}
SEV = ["CRITICAL", "HIGH", "MODERATE", "LOW"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG, "savefig.facecolor": FIG_BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID, "grid.alpha": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "font.size": 11,
})


def _save(fig, name):
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def fig1_hallucination_vs_recall():
    df = pd.read_csv(os.path.join(RESULTS, "triage_metrics.csv")).set_index("method")
    methods = ["Rule-Only", "Standard RAG", "PhysRAG-V2X"]
    hall = [df.loc[m, "hallucinated_action_pct"] for m in methods]
    crit_recall = [100 * (1 - df.loc[m, "critical_fn_rate"]) for m in methods]
    x = np.arange(len(methods)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5))
    b1 = ax.bar(x - w / 2, crit_recall, w, color=C_EM, label="critical recall (%)")
    b2 = ax.bar(x + w / 2, hall, w, color=C_STD, label="hallucinated action rate (%)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", color=FG, fontsize=9)
    _style(ax); ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("percent"); ax.set_ylim(0, 108)
    ax.set_title("Critical recall vs. hallucinated actions\n"
                 "PhysRAG recovers critical incidents with zero hallucinated actions",
                 color=FG)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "fig1_hallucination_vs_recall")


def fig2_confusion():
    df = pd.read_csv(os.path.join(RESULTS, "tickets.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, method in zip(axs, ["Standard RAG", "PhysRAG-V2X"]):
        g = df[df.method == method]
        M = np.zeros((len(SEV), len(SEV)))
        for i, t in enumerate(SEV):
            sub = g[g.true_severity == t]
            for j, p in enumerate(SEV):
                M[i, j] = (sub.pred_severity == p).sum()
        Mn = M / M.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(Mn, cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(SEV))); ax.set_xticklabels(SEV, rotation=30, ha="right")
        ax.set_yticks(range(len(SEV))); ax.set_yticklabels(SEV)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(method, color=MCOLOR[method])
        ax.grid(False)
        for i in range(len(SEV)):
            for j in range(len(SEV)):
                ax.text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                        color="white" if Mn[i, j] < 0.6 else "black", fontsize=9)
    fig.suptitle("Severity confusion under spoofed BSM telemetry (normalised)",
                 color=FG, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig2_regime_confusion_matrix")


def fig3_latency():
    df = pd.read_csv(os.path.join(RESULTS, "latency_breakdown.csv"))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    colors = [C_OURS, C_EM, C_STD]
    bars = ax.bar(df.component, df.latency_ms, color=colors)
    ax.set_yscale("log")
    for b, v in zip(bars, df.latency_ms):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v:.3f} ms", ha="center",
                color=FG, fontsize=9)
    _style(ax)
    ax.set_ylabel("mean latency (ms, log scale)")
    ax.set_title("PhysRAG-V2X per-stage latency decomposition", color=FG)
    _save(fig, "fig3_latency_breakdown")


def fig4_workload():
    df = pd.read_csv(os.path.join(RESULTS, "workload_curve.csv"))
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for m in ["Rule-Only", "Standard RAG", "PhysRAG-V2X"]:
        g = df[df.method == m].sort_values("alert_volume")
        lw = 2.6 if m == "PhysRAG-V2X" else 1.8
        ax.plot(g.alert_volume, g.false_escalations, "-o", ms=4, color=MCOLOR[m], lw=lw, label=m)
    _style(ax)
    ax.set_xlabel("alert volume (incidents processed)")
    ax.set_ylabel("false escalations to operator")
    ax.set_title("Operator-workload reduction under alert-volume surge\n"
                 "(benign / sensor-drift incidents wrongly escalated)", color=FG)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "fig4_operator_workload_reduction")


def main():
    print("Generating publication figures (300 DPI) ...")
    fig1_hallucination_vs_recall()
    fig2_confusion()
    fig3_latency()
    fig4_workload()
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
