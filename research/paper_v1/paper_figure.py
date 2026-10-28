"""K1/K3: paper figure, drawn only from the shortcut-audit receipts."""
import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EVIDENCE = Path('evidence/paper-shortcut-audit-20261025')
INK, MUTED, SURFACE, FAIL = '#0b0b0b', '#52514e', '#fcfcfb', '#d9d8d3'
CLASSES = [('launched', 'calls its own kernel', '#2a78d6'),
           ('decorative', 'decorative kernel', '#eb6834'),
           ('plain', 'plain PyTorch', '#1baf7a')]


def heldout():
    counts = json.loads((EVIDENCE / 'shortcut-audit.json').read_text())['counts']
    rows = []
    for arm, label in [('base', 'Base'), ('C', 'GRPO (C)'), ('SFT', 'RFT (SFT)')]:
        c, n = counts[arm]['correct_by_class'], counts[arm]['attempts']
        rows.append((label, {'launched': c.get('launched', 0) / n,
                             'decorative': c.get('declared_not_launched', 0) / n,
                             'plain': c.get('no_declared_kernel', 0) / n}))
    return rows


def training():
    d = json.loads((EVIDENCE / 'training-shortcut-audit.json').read_text())
    def total(cs):
        t = Counter()
        for c in cs:
            t.update(c)
        return t
    groups = [('Base samples', total(d['base_candidates_layernorm'].values())),
              ('GRPO (C) rollouts', total(c for run, v in d['grpo_rollouts_layernorm'].items() if run.startswith('C-')
                                          for b, c in v['buckets'].items() if b != 'step_0'))]
    rows = []
    for label, t in groups:
        n = t['n']
        rows.append((label, {'launched': t['correct_reachable'] / n,
                             # unreachable kernels did not launch in 153/154 validated cases
                             'decorative': (t['correct_definition_only'] + t['correct_unreachable_other']) / n,
                             'plain': t['correct_no_declared_kernel'] / n}))
    return rows


def panel(ax, rows, title):
    for y, (label, shares) in enumerate(reversed(rows)):
        left = 0.0
        for key, _, color in CLASSES:
            w = shares[key]
            ax.barh(y, w, left=left, height=0.56, color=color, edgecolor=SURFACE, linewidth=2)
            if w >= 0.06:
                ax.text(left + w / 2, y, f'{100 * w:.0f}', ha='center', va='center', color='white',
                        fontsize=9, fontweight='bold')
            left += w
        ax.barh(y, 1 - left, left=left, height=0.56, color=FAIL, edgecolor=SURFACE, linewidth=2)
        ax.text(left + 0.012, y + 0.11, f'{100 * left:.0f}% correct', ha='left', va='center', color=INK, fontsize=9)
        ax.text(left + 0.012, y - 0.15, f'{100 * shares["launched"]:.1f}% via own kernel', ha='left',
                va='center', color=MUTED, fontsize=7.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=INK, fontsize=9.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ['0', '25', '50', '75', '100%'], color=MUTED, fontsize=8)
    ax.set_title(title, loc='left', color=INK, fontsize=10.5, pad=8)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(FAIL)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('paper/figure-decomposition.png'))
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.9), height_ratios=[3, 2], facecolor=SURFACE)
    panel(axes[0], heldout(), 'Held-out Level 2 (5,600 programs): correct programs by observed kernel use')
    panel(axes[1], training(), 'Training task, LayerNorm: correct programs by static class')
    for ax in axes:
        ax.set_facecolor(SURFACE)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CLASSES] + [plt.Rectangle((0, 0), 1, 1, color=FAIL)]
    fig.legend(handles, [l for _, l, _ in CLASSES] + ['incorrect / failed'], loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, 0.035),
               frameon=False, fontsize=8.5, labelcolor=INK)
    fig.text(0.01, 0.008, 'Share of all attempts. Top: profiler attribution. Bottom: static classes, validated against it;'
             ' decorative includes unreachable kernels.',
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0.02, 0.1, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(args.out)


if __name__ == '__main__':
    main()
