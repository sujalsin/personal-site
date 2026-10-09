"""Render saved paired measurements; requires matplotlib, no GPU or network.

From the repository root:
    python scripts/render-attention-residuals-figure.py
    python scripts/render-attention-residuals-figure.py --png /tmp/attention-results.png

The source JSON records notebook provenance. These are saved output measurements,
not regenerated training results. Every comparison uses its own paired baseline.
"""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "editorial/data/attention-residuals-summary.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--png", type=Path)
args = parser.parse_args()

paper, ink, muted, blue, teal, line = (
    "#fafaf8", "#222724", "#626864", "#324d83", "#477567", "#dedfd9"
)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "text.color": ink, "axes.labelcolor": muted,
    "xtick.color": muted, "ytick.color": ink,
    "figure.facecolor": paper, "axes.facecolor": paper,
    "svg.fonttype": "none", "svg.hashsalt": "attention-residuals-results-v1",
})

fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.1), sharex=True)
fig.subplots_adjust(left=.245, right=.965, top=.84, bottom=.19, hspace=.65)
fig.text(.06, .95, "Learning per token and learning against the clock", fontsize=16,
         fontweight="bold", ha="left")
fig.text(.06, .911, "Validation loss difference · AttnRes minus its paired baseline",
         fontsize=11, color=muted)

panels = [
    ("fixed_tokens_validation", "At 40.96M processed tokens"),
    ("fixed_time_validation", "At 1,200 measured training seconds"),
]
for ax, (metric, title) in zip(axes, panels):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(line)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=line, linewidth=.65)
    ax.axvline(0, color=muted, linewidth=1.1, linestyle=(0, (3, 3)))
    ax.set_xlim(-.08, .04)
    ax.set_ylim(-.42, 1.59)
    ax.set_yticks([1, 0], ["Original", "After rewrite"])
    ax.set_xticks([-.08, -.06, -.04, -.02, 0, .02, .04])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:+.2f}".replace("-", "−")))
    ax.tick_params(axis="both", length=0, pad=9, labelbottom=True)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=17)
    for phase, y, color in [("original", 1, blue), ("followup", 0, teal)]:
        stats = data[phase][metric]
        values = stats["values"]
        mean = statistics.mean(values)
        # Catch transcription edits that leave the saved summaries inconsistent.
        if len(values) != len(data["seeds"]) or abs(mean - stats["mean"]) > 1e-12:
            raise ValueError(f"Paired mean does not match saved values: {phase}/{metric}")
        if abs(statistics.stdev(values) - stats["sample_sd"]) > 1e-12:
            raise ValueError(f"Sample SD does not match saved values: {phase}/{metric}")
        ax.scatter(values, [y + .14, y, y - .14], s=39, color=color,
                   edgecolors=paper, linewidth=.7, zorder=3)
        ax.scatter([mean], [y], s=76, marker="D", color=color,
                   edgecolors=paper, linewidth=1.5, zorder=4)
        ax.text(.038, y + .28, f"mean {mean:+.4f}".replace("-", "−"),
                ha="right", va="center", fontsize=10, color=color, fontweight="bold")

axes[-1].set_xlabel("Difference in validation loss (nats/token)", labelpad=12)
legend = [
    Line2D([], [], marker="o", linestyle="none", markersize=5.5, color=muted,
           label="Three paired seeds"),
    Line2D([], [], marker="D", linestyle="none", markersize=7, color=muted,
           label="Mean"),
]
fig.legend(handles=legend, loc="lower left", bbox_to_anchor=(.048, .077),
           frameon=False, ncol=2, fontsize=10, handletextpad=.5, columnspacing=2)
fig.text(.06, .067, "Negative favors AttnRes. Follow-up uses fresh paired baselines.",
         fontsize=10, color=muted)
fig.text(.06, .035, "After rewrite: exploratory validation results; no follow-up test evaluation.",
         fontsize=10, color=muted)

fig.savefig(ROOT / "src/assets/attention-residuals-results.svg", metadata={"Date": None})
if args.png:
    fig.savefig(args.png, dpi=160)
plt.close(fig)
