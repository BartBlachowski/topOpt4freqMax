#!/usr/bin/env python3
"""Diagram figures 13 (first-divergence chain), 14 (causal map), 20 (migration classification).
Status colours carry an icon/label, never colour alone."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sd_common import FIG

SURF, INK, INK2, EDGE = '#fcfcfb', '#0b0b0b', '#52514e', '#c9c8c2'
GOOD, WARN, CRIT, NEUT = '#0ca30c', '#fab219', '#d03b3b', '#f0efec'
plt.rcParams.update({'figure.facecolor': SURF, 'savefig.facecolor': SURF, 'font.size': 9, 'text.color': INK})


def box(ax, x, y, w, h, text, fc=NEUT, ec=EDGE, fs=8.5, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, wrap=True, weight=weight)


def fig13():
    fig, ax = plt.subplots(figsize=(12, 4.2)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 4.2)
    lv = [('D0', 'problem\nconfig', True, 'M1 vs C480: only\ncontroller/stop/runtime\nleaves differ'),
          ('D1', 'K, M,\neigenpairs', True, 'bitwise,\n9 states'), ('D2', 'raw f_sk', True, 'bitwise'),
          ('D3', 'filtered f_sk', True, 'bitwise'), ('D4', 'N, dOff,\nnext-mode', True, 'bitwise'),
          ('D5', '(25) rows', True, 'bitwise'), ('D6', 'inner solve\nsame box', True, 'bitwise\n(box 0.04)'),
          ('D7', 'outer box /\ncontroller', False, 'rho0: box 0.10 vs 0.04\n||drho|| 13.39 vs 4.27'),
          ('D8', 'stopping', False, 'eps-test vs\nterminal E')]
    for i, (code, name, ok, ev) in enumerate(lv):
        x = 0.15 + i * 1.3
        box(ax, x, 2.2, 1.15, 1.2, f'{code}\n{name}', fc=('#e6f4e6' if ok else '#fbe3e3'), ec=(GOOD if ok else CRIT), fs=8)
        ax.text(x + 0.575, 1.95, ('✓ identical' if ok else '✗ FIRST\nDIVERGENCE' if code == 'D7' else '✗ differs'),
                ha='center', va='top', fontsize=7.5, color=(GOOD if ok else CRIT), weight='bold')
        ax.text(x + 0.575, 1.45, ev, ha='center', va='top', fontsize=7, color=INK2)
    ax.text(0.15, 3.85, 'First divergence under the maximally matched pairing M1 (source code with the target material law vs target C480), evaluated at the shared state rho0',
            fontsize=10, weight='bold')
    ax.text(0.15, 0.45, 'M0 (native) additionally differs at D0 in the material law, but that operator is exactly zero while every element has rho > 0.1: '
            '\nit activates at outer iteration 6 on the source trajectory (P-prefix test: S480 = M1 bitwise for iterations 1-5).', fontsize=8.2, color=INK2)
    ax.text(0.15, -0.05, 'A first divergence is a location, not a cause (preregistration §9).', fontsize=8.2, color=INK2, style='italic')
    fig.savefig(FIG / 'fig13_first_divergence.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def fig14():
    fig, ax = plt.subplots(figsize=(12, 6.2)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 6.2)
    ax.text(0.1, 5.95, 'Causal map at 480×60 — factors (left) → observed behaviour (right); edge label = evidence level', fontsize=10, weight='bold')
    F = {'ped': (0.2, 4.6, 'Pedersen low-density stiffness\n(rho/100 below 0.1)'), 'mass': (0.2, 3.6, 'linear mass eq.(2)\nvs eq.(4b)'),
         'box': (0.2, 2.6, 'adaptive per-element box\n(0.10 ceiling, x1.2 / x0.7)'), 'stop': (0.2, 1.6, 'eps-test, no guards\nvs terminal stage-exhaustion'),
         'same': (0.2, 0.3, 'IDENTICAL: FE, eigs, gradients, filter,\nmultiplicity, rows, inner MMA')}
    O = {'spike': (8.2, 4.6, 'no localized-mode spikes\n(M1: 11; A2@240: 13; C3@800: killed)'),
         'mnd': (8.2, 3.5, 'low final M_nd 0.131\n(M1 0.285, C480 0.263)'),
         'rate': (8.2, 2.4, 'fast early sharpening\n(M_nd@64: 0.29 vs 0.51)'),
         'term': (8.2, 1.3, 'natural termination\n(S480 @112; M1 falsely @64)')}
    for k, (x, y, t) in F.items():
        box(ax, x, y, 3.0, 0.8, t, fc=('#e8f0fb' if k != 'same' else NEUT))
    for k, (x, y, t) in O.items():
        box(ax, x, y, 3.4, 0.8, t, fc='#eef7f2')
    E = [('ped', 'spike', 'STRONG (single factor + same-state)', CRIT, .5), ('ped', 'mnd', 'STRONG (single factor)', CRIT, .72),
         ('mass', 'mnd', 'MODERATE against (P1/P2 @240)', WARN, .78), ('box', 'rate', 'STRONG (first divergence)', CRIT, .5),
         ('box', 'spike', 'interaction: spikes only with SIMP/4b', WARN, .22), ('stop', 'term', 'STRONG (mechanism, M1 false stop)', CRIT, .72),
         ('box', 'term', 'STRONG (box collapse drives eps)', CRIT, .3), ('same', 'mnd', 'IDENTICAL → not a cause', INK2, .12)]
    for a, b, lab, c, tt in E:
        xa, ya = F[a][0] + 3.0, F[a][1] + 0.4; xb, yb = O[b][0], O[b][1] + 0.4
        ax.annotate('', xy=(xb, yb), xytext=(xa, ya), arrowprops=dict(arrowstyle='->', color=c, lw=1.4 if c != INK2 else 0.9, ls='-' if c != INK2 else ':'))
        ax.text(xa + tt * (xb - xa), ya + tt * (yb - ya) + 0.08, lab, fontsize=7.2, color=INK2, ha='center',
                bbox=dict(fc=SURF, ec='none', pad=0.5))
    ax.text(0.1, -0.05, 'Not isolated: Pedersen × three-rung ladder (the missing cell). Filter non-conservativity and MMA attenuation are shared, so they cannot explain the difference.', fontsize=8, color=INK2)
    fig.savefig(FIG / 'fig14_causal_map.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def fig20():
    rows = [('Pedersen stiffness (stiffnessInterpolation + struct in assemble2D/genGrad)', 'PROMOTE AS NEW NAMED PRESET'),
            ('Linear mass eq.(2) selection', 'PROMOTE AS NEW NAMED PRESET'),
            ('Adaptive per-element box (limit.m adaptive, vector box in olhoffSolve)', 'PROMOTE AS NEW NAMED PRESET'),
            ('duOlhoffAdaptivePedersen realization + eps stop without guards', 'PROMOTE AS NEW NAMED PRESET'),
            ('eigSolve opts pass-through; res.aux; hist.move = max box', 'PROMOTE NOW'),
            ('tOuter timing instrumentation (target)', 'PROMOTE NOW'),
            ('Stage-exhaustion controller (target exhaustion.m etc.)', 'KEEP HISTORICAL ONLY'),
            ('Frozen preset duOlhoffFrozenM4 / duOlhoffFixedPenaltySensitivityFiltered', 'KEEP HISTORICAL ONLY'),
            ('duOlhoffAdaptiveMove (SIMP/4b + adaptive box)', 'KEEP HISTORICAL ONLY'),
            ('duOlhoffOuterAsymptotes + asymptoteHistory=outer', 'DO NOT PROMOTE'),
            ('boxInactiveFraction / settledWindow / move.initial=Inf options', 'DO NOT PROMOTE'),
            ('Plan Phase 3: replace the production preset in place', 'DO NOT PROMOTE'),
            ('Uncommitted source §7: rhomin 1e-7, Nmax 2, Pedersen for all methods', 'DO NOT PROMOTE'),
            ('Pedersen under the three-rung ladder (would it repair the old preset?)', 'NEEDS CAUSAL TEST'),
            ('Plan Phase 6.1 common radius / 6.2 Proposed low-density law', 'NEEDS SEPARATE AUTHORIZATION')]
    col = {'PROMOTE NOW': ('#e6f4e6', GOOD, '▲'), 'PROMOTE AS NEW NAMED PRESET': ('#e8f0fb', '#2a78d6', '◆'),
           'KEEP HISTORICAL ONLY': (NEUT, INK2, '■'), 'DO NOT PROMOTE': ('#fbe3e3', CRIT, '✗'),
           'NEEDS CAUSAL TEST': ('#fdf1d6', '#c98500', '?'), 'NEEDS SEPARATE AUTHORIZATION': ('#fdf1d6', '#c98500', '‖')}
    fig, ax = plt.subplots(figsize=(12, 6.6)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, len(rows) + 1.2)
    ax.text(0.05, len(rows) + 0.6, 'Migration classification (Part 19) — gate: OLHOFFCURRENT_MIGRATION_READY_WITH_NAMED_FORMULATION_SPLIT', fontsize=10, weight='bold')
    for i, (item, cls) in enumerate(rows):
        y = len(rows) - 1 - i
        fc, ec, ic = col[cls]
        ax.text(0.05, y + 0.35, item, fontsize=8.5, va='center')
        box(ax, 7.6, y + 0.05, 4.3, 0.6, f'{ic}  {cls}', fc=fc, ec=ec, fs=8.3, weight='bold')
    fig.savefig(FIG / 'fig20_migration_classification.png', dpi=160, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    fig13(); fig14(); fig20(); print('diagrams written')
