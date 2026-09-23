"""Boxplot: days-to-resolve per proposed QC category, two views.

    python plot_resolution_boxplot.py --input durations.csv --output chart.png

Top panel: operator-conservative view (pipeline-default jump exclusions —
every jump is currently unreviewed, so all are excluded). Bottom panel: raw
view (no jump exclusions). Categories on x (sorted by median, descending),
days on y. Single validated series hue (#2a78d6 on #fcfcfb — dataviz
validator: all checks pass). Whiskers 1.5xIQR, outliers hidden.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = Path(__file__).resolve().parent / "resolution_times_by_category.csv"
OUT = Path(__file__).resolve().parent / "resolution_time_by_category_boxplot.png"

SERIES = "#2a78d6"
INK, INK2, MUTED, RULE = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
GRID, SURFACE, PAGE = "#e1e0d9", "#fcfcfb", "#f9f9f7"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def draw_panel(fig, ax, sub, heading, note):
    order = (sub.groupby("category")["days_to_resolve"].median()
             .sort_values(ascending=False).index.tolist())
    data = [sub.loc[sub["category"] == c, "days_to_resolve"].values
            for c in order]
    ns = [len(d) for d in data]
    medians = [pd.Series(d).median() for d in data]

    bp = ax.boxplot(
        data, positions=range(len(order)), widths=0.52, patch_artist=True,
        showfliers=False, whis=1.5, zorder=3,
        boxprops=dict(facecolor="#2a78d62e", edgecolor=SERIES, linewidth=1.3),
        whiskerprops=dict(color=INK2, linewidth=1.0),
        capprops=dict(color=INK2, linewidth=1.0),
        medianprops=dict(color=INK, linewidth=1.8),
    )
    ymax = max(max(w.get_ydata().max() for w in bp["whiskers"]), 1)
    for pos, med in zip(range(len(order)), medians):
        ax.text(pos + 0.34, max(med, 0.022 * ymax), f"{med:.0f} d",
                va="center", ha="left", fontsize=8.5, color=INK2, zorder=4)

    labels = [textwrap.fill(c.replace("/", "/ "), 16).replace("/ ", "/")
              + f"\nn={n:,}" for c, n in zip(order, ns)]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK2)
    ax.set_ylabel("days to resolve", fontsize=9, color=INK2)
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=8.5, colors=INK2, length=0, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.8)
    ax.margins(x=0.04)
    ax.set_ylim(bottom=0)

    pos = ax.get_position()
    fig.text(0.035, pos.y1 + 0.035, heading, ha="left", va="baseline",
             fontsize=10.5, fontweight="600", color=INK)
    fig.text(0.035, pos.y1 + 0.014, note, ha="left", va="baseline",
             fontsize=8.5, color=INK2)


def main(input_path=SRC, output_path=OUT):
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    df = pd.read_csv(input_path)
    cons = df[df["view"] == "conservative"]
    raw = df[df["view"] == "raw"]

    fig, axes = plt.subplots(
        2, 1, figsize=(11.4, 9.6), dpi=200, facecolor=PAGE,
        gridspec_kw={"hspace": 0.52, "left": 0.075, "right": 0.985,
                     "top": 0.845, "bottom": 0.075})

    draw_panel(
        fig, axes[0], cons,
        f"Operator-conservative view (pipeline default)  —  "
        f"{len(cons):,} resolved flags",
        "Jump exclusions applied. Every add/remove jump is still unreviewed "
        "in jumps_{network}.xlsx, so ALL are excluded by default — only "
        "organic, sub-threshold resolutions remain.")
    draw_panel(
        fig, axes[1], raw,
        f"Raw view (no jump exclusions)  —  {len(raw):,} resolved flags",
        "Every closed eligible entry counts as resolved, including bulk "
        "disappearances (check removals / rewordings / tracker cutovers), "
        "which can fabricate short 'resolutions'.")

    fig.text(0.035, 0.965, "Days to resolve QC flags, by proposed category",
             ha="left", va="baseline", fontsize=15, fontweight="600",
             color=INK)
    fig.text(0.035, 0.938,
             "8/19 tracker-history run, clinical forms, PRESCIENT + PRONET "
             "pooled. Duration = last revision the flag was present minus "
             "first seen (lower bound: this run's history predates the "
             "Resolution_observed column). Boxes span Q1\u2013Q3, whiskers "
             "1.5\u00d7IQR, outliers hidden, medians labeled.",
             ha="left", va="baseline", fontsize=8.5, color=INK2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=PAGE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV produced by compute_resolution_times.py")
    parser.add_argument("--output", type=Path,
                        default=Path("resolution_time_by_category_boxplot.png"),
                        help="output figure (default: %(default)s)")
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    main(args.input, args.output)


if __name__ == "__main__":
    cli()
