"""Render the article figure from transcribed study aggregates; needs matplotlib.

Run from the repository root: python scripts/render-booking-figure.py
Optional review image: python scripts/render-booking-figure.py --png /tmp/booking-study-results.png
No model generation, candidate execution, or network access is involved.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'editorial/data/booking-study-summary.json').read_text())
args = argparse.ArgumentParser()
args.add_argument('--png', type=Path)
options = args.parse_args()

paper, ink, muted, blue, line = '#fafaf8', '#222724', '#626864', '#324d83', '#dedfd9'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': ink, 'axes.labelcolor': muted, 'xtick.color': ink,
    'ytick.color': muted, 'figure.facecolor': paper, 'axes.facecolor': paper,
    'svg.fonttype': 'none', 'svg.hashsalt': 'booking-study-results-v1',
})
fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.3))
fig.subplots_adjust(left=.13, right=.95, top=.90, bottom=.105, hspace=.87)
x = np.arange(3)
for ax in axes:
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(line)
    ax.tick_params(axis='both', length=0, pad=8)
    ax.grid(axis='y', color=line, linewidth=.7, zorder=0)
    ax.set_xticks(x, data['conditions'])
    ax.set_xlim(-.5, 2.5)

ax = axes[0]
ax.set_title('1. Grade the same saved programs', loc='left', fontweight='bold', fontsize=14, pad=34)
ax.text(0, 1.055, '2,048 draws · every grader retained all 440 audit-passing draws', transform=ax.transAxes, color=muted, fontsize=10)
ax.bar(x, data['accepts_audit_failing'], width=.36, color=[blue, '#9a6447', blue], zorder=3)
for pos, count in zip(x, data['accepts_audit_failing']):
    ax.text(pos, count + 1.7, str(count), ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylim(0, 46)
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_ylabel('Audit-failing draws accepted', labelpad=13)

ax = axes[1]
ax.set_title('2. Evaluate policies trained with each grader', loc='left', fontweight='bold', fontsize=14, pad=34)
ax.text(0, 1.055, 'Four paired seeds · gray: each seed · blue diamonds: mean', transform=ax.transAxes, color=muted, fontsize=10)
values = np.array(data['final_audit_passes_by_seed'], dtype=float) / data['final_draws_per_policy'] * 100
for row in values:
    ax.plot(x, row, color='#9caaa3', alpha=.8, linewidth=1.2, marker='o', markersize=5, zorder=2)
means = values.mean(axis=0)
ax.scatter(x, means, color=blue, marker='D', s=65, edgecolors=paper, linewidths=1, zorder=4)
for pos, val in zip(x, means):
    ax.text(pos, 4.0, f'{val:.2f}%', ha='center', color=blue, fontweight='bold', fontsize=12)
ax.set_ylim(0, 40)
ax.set_yticks([0, 10, 20, 30, 40], ['0%', '10%', '20%', '30%', '40%'])
ax.set_ylabel('Pass all 192 audit cases', labelpad=13)
fig.text(.13, .025, 'Final policy comparison: 128 draws per policy, 512 per condition.\nGrader improvement did not establish a training benefit.', color=muted, fontsize=10, linespacing=1.5)
fig.savefig(ROOT / 'src/assets/booking-study-results.svg', metadata={'Date': None})
if options.png:
    fig.savefig(options.png, dpi=150)
plt.close(fig)
