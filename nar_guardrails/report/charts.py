"""Report charts (matplotlib, Agg). Palette follows the dataviz skill's reference instance:
categorical slots 1/2 (blue/orange), a single-hue blue sequential ramp, recessive grid, ink in text tokens."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LOCAL, REMOTE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _empty(out_png: Path, msg: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2), facecolor=SURFACE)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=INK2)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_png)


def scatter_cost_latency(lb: pd.DataFrame, out_png: Path) -> Path:
    df = lb.dropna(subset=["p50_ms", "fpr_at_r95"]) if len(lb) else lb
    df = df[df["p50_ms"] > 0] if len(df) else df
    if df.empty:
        return _empty(out_png, "no detector has latency and a dev-chosen 95% recall threshold")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    cost = df["per_1k_usd"].fillna(0).astype(float)
    sizes = 50 + 400 * (cost / max(cost.max(), 1e-9))
    colors = np.where(df["kind"] == "remote", REMOTE, LOCAL)
    ax.scatter(df["p50_ms"], df["fpr_at_r95"], s=sizes, c=colors, alpha=0.85, edgecolor=SURFACE, linewidth=2, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["detector"], (r["p50_ms"], r["fpr_at_r95"]), textcoords="offset points", xytext=(7, 6),
                    fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(-0.02, max(0.1, float(df["fpr_at_r95"].max()) * 1.15))
    ax.set_xlabel("p50 latency per item (ms, log scale)", color=INK2)
    ax.set_ylabel("false-positive rate at the dev-chosen 95 % recall threshold", color=INK2)
    ax.set_title("Cost, latency and false-positive rate (marker size = $ per 1k items)", color=INK, fontsize=10, loc="left")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markeredgecolor=SURFACE, markersize=8, label=n)
                       for n, c in (("local", LOCAL), ("remote", REMOTE))],
              frameon=False, fontsize=8, labelcolor=INK2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_png)


def _draw_panel(ax, fig, data: np.ndarray, xlabels: list[str], ylabels: list[str], cbar_label: str | None) -> None:
    im = ax.imshow(data, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            v = data[y, x]
            if not np.isnan(v):
                ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.45 else INK)
    if cbar_label:
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(cbar_label, color=INK2, fontsize=8)
        cb.ax.tick_params(colors=INK2, labelsize=7)
        cb.outline.set_visible(False)


def scenario_heatmap(sc: pd.DataFrame, out_png: Path) -> Path:
    fpr_cols = [c for c in sc.columns if str(c).endswith("_fpr")]
    cols = [c for c in sc.columns if c not in ("n", "metric") and c not in fpr_cols]
    if sc.empty or not cols:
        return _empty(out_png, "no scenario results")
    rows = [f"{i} ({m}, n={n})" for i, m, n in zip(sc.index, sc["metric"], sc["n"])]
    panels = [(sc[cols].astype(float).values, cols, "recall (malicious) / FPR (benign)")]
    mixed = [c for c in fpr_cols if sc[c].notna().any()]
    if mixed:
        panels.append((sc[mixed].astype(float).values, [c[:-4] for c in mixed], "FPR (benign part of mixed)"))
    widths = [max(len(c), 1) + 1.5 for _, c, _ in panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.5 + 0.9 * sum(len(c) for _, c, _ in panels), max(3.0, 1.6 + 0.4 * len(sc))),
                             facecolor=SURFACE, gridspec_kw={"width_ratios": widths}, squeeze=False)
    for k, (ax, (data, xl, lab)) in enumerate(zip(axes[0], panels)):
        ax.set_facecolor(SURFACE)
        _draw_panel(ax, fig, data, xl, rows if k == 0 else [""] * len(rows), lab)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return Path(out_png)
