#!/usr/bin/env python3
"""Generate publication-style figures for the RL workshop draft."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
O2O_DIR = ROOT / "refine-logs" / "remote-results" / "o2o_p0_eval20_20260514" / "results"
QGATE_DIR = ROOT / "refine-logs" / "remote-results" / "o2o_p1_qgate_eval20_20260514" / "results"

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#6E6E6E",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def add_box(ax, xy, wh, text, fc, ec="#2B2B2B", fontsize=9, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.15,
        color="#1A1A1A",
    )
    return patch


def add_arrow(ax, start, end, color="#333333", rad=0.0, lw=1.4):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(arrow)
    return arrow


def fig_project_story() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.95,
        "Low-quality replay data: three constraints, one O2O question",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.895,
        "Which constraint helps offline learning, and which must be relaxed online?",
        ha="center",
        va="center",
        fontsize=9,
        color="#4A4A4A",
    )

    add_box(ax, (0.05, 0.52), (0.20, 0.16), "Offline replay\nD4RL medium-replay", "#F5F5F5", weight="bold")
    add_box(ax, (0.40, 0.52), (0.20, 0.16), "Offline\npretraining", "#F5F5F5", weight="bold")
    add_box(ax, (0.75, 0.52), (0.20, 0.16), "Online\nfine-tuning", "#F5F5F5", weight="bold")
    add_arrow(ax, (0.25, 0.60), (0.40, 0.60))
    add_arrow(ax, (0.60, 0.60), (0.75, 0.60))

    add_box(ax, (0.36, 0.37), (0.28, 0.08), "Compare constraint families\nunder the same protocol", "#FFFFFF", ec="#777777", fontsize=8.5, weight="bold")
    add_arrow(ax, (0.50, 0.52), (0.50, 0.45), "#777777", lw=1.2)

    tracks = [
        ("A", "Value\nconservatism\nCQL / Cal-QL", OKABE_ITO["blue"], 0.07),
        ("B", "Non-conservative\ncontrast\nPPO / SAC / TD3", OKABE_ITO["orange"], 0.37),
        ("C", "Policy regularization\ntrusted actions\nTD3+BC / SSAR / ATLAS", OKABE_ITO["green"], 0.67),
    ]
    for letter, text, color, x in tracks:
        add_box(ax, (x, 0.14), (0.26, 0.17), text, "#FFFFFF", ec=color, fontsize=8.5, weight="bold")
        ax.text(x + 0.015, 0.29, letter, ha="left", va="center", color=color, fontsize=12, fontweight="bold")
        add_arrow(ax, (0.50, 0.37), (x + 0.13, 0.31), color=color, lw=1.0, rad=0.08 if x < 0.3 else (-0.08 if x > 0.5 else 0.0))

    ax.text(
        0.50,
        0.07,
        "Merge rule: same env, seed, steps, eval episodes, final/best normalized score, and curve/log path",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4A4A4A",
    )
    save(fig, "fig1_project_story")


def fig_atlas_mechanism() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "ATLAS distills trusted-action information after the teacher cache exists", ha="center", va="center", fontsize=12, fontweight="bold")

    add_box(ax, (0.04, 0.61), (0.20, 0.16), "SSAR / IQL-qv\nteacher cache", "#EAF2FA", ec=OKABE_ITO["blue"], weight="bold")
    add_box(ax, (0.30, 0.61), (0.19, 0.16), "Export labels\nQ(s,a) - V(s)", "#F7F3E8", ec=OKABE_ITO["orange"], weight="bold")
    add_box(ax, (0.55, 0.61), (0.19, 0.16), "ATLAS selector\n$g_\\phi(s,a)$", "#EAF5EF", ec=OKABE_ITO["green"], weight="bold")
    add_box(ax, (0.79, 0.61), (0.17, 0.16), "Weighted\nTD3+BC", "#F7EEF7", ec=OKABE_ITO["purple"], weight="bold")

    add_arrow(ax, (0.24, 0.69), (0.30, 0.69), OKABE_ITO["blue"])
    add_arrow(ax, (0.49, 0.69), (0.55, 0.69), OKABE_ITO["orange"])
    add_arrow(ax, (0.74, 0.69), (0.79, 0.69), OKABE_ITO["green"])

    add_box(ax, (0.08, 0.29), (0.25, 0.15), "Expensive once\npreselection cost", "#FFFFFF", ec="#888888")
    add_box(ax, (0.38, 0.29), (0.25, 0.15), "Reusable signal\naligned labels matter", "#FFFFFF", ec="#888888")
    add_box(ax, (0.68, 0.29), (0.25, 0.15), "Online caveat\nover-constraint risk", "#FFFFFF", ec="#888888")

    add_arrow(ax, (0.14, 0.61), (0.20, 0.44), "#777777", rad=0.15)
    add_arrow(ax, (0.64, 0.61), (0.50, 0.44), "#777777", rad=-0.15)
    add_arrow(ax, (0.87, 0.61), (0.81, 0.44), "#777777", rad=-0.15)

    ax.text(0.5, 0.11, "Claim boundary: post-cache / amortized cheaper; not a from-scratch SOTA claim", ha="center", va="center", fontsize=9, color="#4A4A4A")
    save(fig, "fig2_atlas_mechanism")


def fig_offline_results() -> None:
    labels = ["TD3+BC\n50k", "ReBRAC-lite\n50k", "SSAR\n50k", "CQL\n50k", "ATLAS\n100k", "ATLAS\nshuffled"]
    scores = np.array([22.43, 34.48, 38.56, 39.81, 69.97, 18.78])
    colors = [
        "#BDBDBD",
        OKABE_ITO["sky"],
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["green"],
        OKABE_ITO["vermillion"],
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_ylabel("Normalized score")
    ax.set_title("C-line offline evidence on hopper-medium-replay-v2", pad=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 80)
    ax.grid(axis="y", color="#E0E0E0", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 1.4, f"{score:.1f}", ha="center", va="bottom", fontsize=8)
    ax.annotate(
        "teacher-label\nalignment matters",
        xy=(5, 18.78),
        xytext=(4.45, 50),
        arrowprops=dict(arrowstyle="-|>", color=OKABE_ITO["vermillion"], lw=1.2),
        ha="center",
        fontsize=8.5,
        color="#333333",
    )
    ax.text(0.01, -0.20, "Scores are exploration-stage single-seed results; SSAR/ATLAS use different budgets as labeled.", transform=ax.transAxes, fontsize=7.5, color="#555555")
    save(fig, "fig3_offline_results")


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fig_o2o_curve() -> None:
    runs = [
        ("TD3+BC release", O2O_DIR / "td3_bc_o2o_eval20_decay_hopper-medium-replay-v2_seed0.jsonl", OKABE_ITO["blue"], "o", "-"),
        ("ATLAS release", O2O_DIR / "atlas_o2o_eval20_decay_hopper-medium-replay-v2_seed0.jsonl", OKABE_ITO["green"], "^", "-"),
        ("ATLAS q-gate", QGATE_DIR / "atlas_o2o_eval20_qgate_fixed_hopper-medium-replay-v2_seed0.jsonl", OKABE_ITO["orange"], "P", "-"),
        ("Random trust", O2O_DIR / "random_subset_iqlqv_o2o_eval20_decay_hopper-medium-replay-v2_seed0.jsonl", OKABE_ITO["gray"], "v", "--"),
        ("SSAR/IQL fixed", O2O_DIR / "ssar_iqlqv_o2o_eval20_fixed_hopper-medium-replay-v2_seed0.jsonl", OKABE_ITO["vermillion"], "D", "--"),
    ]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.45, 1.0]})
    for label, path, color, marker, linestyle in runs:
        rows = read_jsonl(path)
        steps = np.array([r["step"] for r in rows]) / 1000
        scores = np.array([r["normalized_score"] for r in rows])
        ax.plot(steps, scores, label=label, color=color, marker=marker, markersize=3.5, linewidth=1.45, linestyle=linestyle)

    ax.axvline(50, color="#777777", linestyle="--", linewidth=1.0)
    ax.text(50.4, 93, "online starts", fontsize=8, color="#555555", va="top")
    ax.set_xlabel("Training steps (k)")
    ax.set_ylabel("Normalized score")
    ax.set_title("Eval20 O2O curves", pad=8, fontweight="bold")
    ax.set_xlim(9, 61)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#E0E0E0", linestyle="--", linewidth=0.7)
    ax.legend(frameon=False, ncol=1, loc="upper left")

    summary = [
        ("TD3+BC\nrelease", 22.20, 40.06, OKABE_ITO["blue"]),
        ("ATLAS\nrelease", 46.70, 37.50, OKABE_ITO["green"]),
        ("ATLAS\nq-gate", 46.70, 48.41, OKABE_ITO["orange"]),
        ("SSAR/IQL\nfixed", 50.71, 38.61, OKABE_ITO["vermillion"]),
    ]
    x = np.arange(len(summary))
    w = 0.34
    offline = [s[1] for s in summary]
    online = [s[2] for s in summary]
    colors = [s[3] for s in summary]
    ax2.bar(x - w / 2, offline, width=w, color="#D9D9D9", edgecolor="#333333", linewidth=0.6, label="offline")
    ax2.bar(x + w / 2, online, width=w, color=colors, edgecolor="#333333", linewidth=0.6, label="online final")
    for i, (off, on) in enumerate(zip(offline, online)):
        ax2.plot([i - w / 2, i + w / 2], [off, on], color="#666666", linewidth=0.9)
        ax2.text(i + w / 2, on + 1.4, f"{on:.1f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in summary])
    ax2.set_ylim(0, 60)
    ax2.set_title("Offline endpoint vs online final", pad=8, fontweight="bold")
    ax2.grid(axis="y", color="#E0E0E0", linestyle="--", linewidth=0.7)
    ax2.legend(frameon=False, loc="upper right")

    fig.text(0.02, -0.02, "Setting: hopper-medium-replay-v2, seed0, 50k offline + 10k online, eval20. A Q-filtered teacher gate improves ATLAS seed0, but seed1 checks show high O2O variance.", fontsize=7.5, color="#555555")
    save(fig, "fig4_o2o_curve")


def write_latex_includes() -> None:
    text = r"""\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{../figures/fig1_project_story.pdf}
\caption{Project framing. The final report compares three constraint families under one offline-to-online question.}
\label{fig:project-story}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{../figures/fig2_atlas_mechanism.pdf}
\caption{ATLAS mechanism. ATLAS distills trusted-action labels from a cached SSAR/IQL-qv teacher and uses them for weighted TD3+BC-style learning.}
\label{fig:atlas-mechanism}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{../figures/fig3_offline_results.pdf}
\caption{C-line offline evidence on \texttt{hopper-medium-replay-v2}. Aligned ATLAS labels improve the offline endpoint, while shuffled labels collapse.}
\label{fig:offline-results}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{../figures/fig4_o2o_curve.pdf}
\caption{P0 offline-to-online eval20 panel. Stronger teacher labels improve offline endpoints but do not improve final online score under the current online regularizer.}
\label{fig:o2o-curve}
\end{figure}
"""
    (FIG_DIR / "latex_includes.tex").write_text(text)


def main() -> None:
    setup_style()
    fig_project_story()
    fig_atlas_mechanism()
    fig_offline_results()
    fig_o2o_curve()
    write_latex_includes()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
