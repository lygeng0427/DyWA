#!/usr/bin/env python3
"""Publication figure: the joint action+track diffusion pipeline with a
meta-learned dynamics belief -- METHOD ONLY.

    python3 _fig_joint_pipeline_clean.py [outdir]

The annotated twin is `_fig_joint_pipeline.py`, which carries the measured
caveats, the pick-bar correspondence and the traps that fail silently. Keep
that one for working discussions; this one states the method and nothing else,
for a paper or a talk.

Differences, deliberate:
  * no cautionary text and no commentary on the reference implementation
  * the belief reaches both models through FiLM alone -- no concat-q path and
    no zero-init output residual
  * five panels instead of six; the correspondence table is gone

House style (figures4papers): Arial/Helvetica fallback stack, spines off
(schematic, not a data plot), the blue / green / red / neutral palette, 300 dpi
PNG plus vector PDF and SVG with `svg.fonttype='none'`.

    blue    the two models
    green   the belief path, and the ground truth that anchors the inner loss
    red     the action modality, and the meta-gradient
    neutral tokens and observations carrying no modality
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

P = {'blue_main': '#0F4D92', 'blue_secondary': '#3775BA', 'blue_pale': '#E3EBF5',
     'green_1': '#DDF3DE', 'green_2': '#AADCA9', 'green_3': '#8BCF8B',
     'red_1': '#F6CFCB', 'red_2': '#E9A6A1', 'red_strong': '#B64342',
     'neutral': '#CFCECE', 'gray_mid': '#767676', 'gray_dark': '#4D4D4D'}
GREEN_DK = '#2E7D46'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 16,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'legend.frameon': False,
    'svg.fonttype': 'none',
})


def box(ax, x, y, w, h, label, sub=None, fc='white', ec=P['blue_main'],
        lw=1.8, fs=10.5, tc=None, bold=True, ls='-', r=0.014):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=%f' % r,
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    tc = tc or ('white' if fc in (P['blue_main'], P['blue_secondary'], GREEN_DK)
                else '#111111')
    ax.text(x + w / 2, y + (h * 0.62 if sub else h * 0.5), label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=tc, zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center',
                fontsize=fs - 2.2, color=tc, alpha=.9, zorder=4)


def arrow(ax, p0, p1, color='#333333', lw=1.8, ls='-', rad=0.0, z=2, head=10):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head,
                                 lw=lw, ls=ls, color=color, zorder=z,
                                 connectionstyle='arc3,rad=%f' % rad,
                                 shrinkA=2, shrinkB=3))


def panel(ax, tag, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.0, 1.04, tag, fontsize=15, fontweight='bold', va='bottom')
    ax.text(0.035, 1.045, title, fontsize=12, va='bottom', color='#333333')


# ---------------------------------------------------------------------------
def panel_a(ax):
    panel(ax, 'a', 'Inner loop  —  the belief is fitted from elapsed motion')

    ax.text(0.0, 0.955,
            'per support window:  DINOv2 391x64  ·  patch centres 391x3  ·  EEF pose 7\n'
            'and the executed action chunk 16x20',
            fontsize=9.6, va='top', color=P['gray_mid'], linespacing=1.7)

    box(ax, 0.03, 0.495, 0.42, 0.185, 'Track Transformer',
        'predicts 391 x 16 x 3', fc=P['blue_main'], ec=P['blue_main'], fs=11.5)
    box(ax, 0.55, 0.495, 0.42, 0.185, 'GT 3D track', 'from the simulator',
        fc=P['green_2'], ec=GREEN_DK, fs=11.5)

    box(ax, 0.23, 0.250, 0.52, 0.160, 'object-only L2',
        'moving object patches, in mm', fc=P['green_1'], ec=GREEN_DK, fs=11)
    arrow(ax, (0.24, 0.495), (0.40, 0.415), P['blue_main'], 1.9)
    arrow(ax, (0.76, 0.495), (0.59, 0.415), GREEN_DK, 1.9)

    box(ax, 0.03, 0.055, 0.155, 0.115, 'q', 'belief, 8-d',
        fc=GREEN_DK, ec=GREEN_DK, fs=13)
    arrow(ax, (0.23, 0.288), (0.188, 0.145), P['red_strong'], 1.9, rad=0.22)
    arrow(ax, (0.108, 0.172), (0.125, 0.492), GREEN_DK, 2.1)
    ax.text(0.143, 0.355, 'FiLM', fontsize=9.8, color=GREEN_DK,
            style='italic', ha='left', va='center')

    ax.text(0.275, 0.150,
            r'$q \;\leftarrow\; q \;-\; \alpha \cdot g/\|g\|$'
            '           g = dL/dq,  K steps',
            fontsize=12, va='center', color='#111111')
    ax.text(0.275, 0.060,
            'alpha is learned;  the step is unit-normalised, so the belief\n'
            'travels exactly K x alpha and all of it is direction.',
            fontsize=9.6, va='center', color=P['gray_mid'], linespacing=1.7)


def panel_b(ax):
    panel(ax, 'b', 'Outer loop  —  one DiT denoises the action and the tracks')

    box(ax, 0.0, 0.660, 0.30, 0.200, 'Context encoder',
        '[CLS, state, 391 patches]', fc=P['blue_main'], ec=P['blue_main'],
        fs=10.8)
    box(ax, 0.0, 0.400, 0.30, 0.125, 'FiLM on every FFN', 'cond_dim = z_dim',
        fc=P['green_2'], ec=GREEN_DK, fs=10.5)
    box(ax, 0.0, 0.205, 0.13, 0.110, 'q', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    arrow(ax, (0.065, 0.317), (0.065, 0.397), GREEN_DK, 2.0)
    arrow(ax, (0.15, 0.528), (0.15, 0.657), GREEN_DK, 2.0)
    arrow(ax, (0.302, 0.462), (0.358, 0.485), GREEN_DK, 2.0)
    arrow(ax, (0.302, 0.760), (0.358, 0.730), P['blue_main'], 2.2)

    ax.add_patch(FancyBboxPatch((0.36, 0.300), 0.40, 0.600,
                                boxstyle='round,pad=0,rounding_size=0.018',
                                fc=P['blue_pale'], ec=P['blue_main'], lw=1.8,
                                zorder=1))
    ax.text(0.56, 0.865, 'Joint DiT  ·  self-attention only',
            fontsize=10.6, ha='center', va='center',
            fontweight='bold', color=P['blue_main'], zorder=4)
    toks = [(0.740, '16  noisy ACTION tokens', P['red_1'], P['red_strong']),
            (0.635, '391  fused TRACK tokens', '#D6E3F2', P['blue_secondary']),
            (0.530, '1  state token', 'white', P['gray_mid']),
            (0.425, '1  diffusion-timestep token', 'white', P['gray_mid'])]
    for y, lab, fc, ec in toks:
        box(ax, 0.378, y, 0.364, 0.082, lab, None, fc=fc, ec=ec, lw=1.5,
            fs=9.6)
    ax.text(0.56, 0.360, '409 tokens  ·  one shared timestep',
            fontsize=9.2, ha='center', va='center', color=P['blue_main'],
            style='italic', zorder=4)

    box(ax, 0.80, 0.705, 0.20, 0.110, 'action head', '16 x 20',
        fc=P['red_1'], ec=P['red_strong'], fs=9.8)
    box(ax, 0.80, 0.550, 0.20, 0.110, 'track head', '391 x 16 x 3',
        fc='#D6E3F2', ec=P['blue_secondary'], fs=9.8)
    arrow(ax, (0.762, 0.750), (0.796, 0.760), P['red_strong'], 1.9)
    arrow(ax, (0.762, 0.640), (0.796, 0.605), P['blue_secondary'], 1.9)

    ax.text(0.36, 0.215,
            r'$L_{\mathrm{out}} \;=\; \mathrm{MSE}(a) \;+\; \lambda\,'
            r'\Sigma w\,\mathrm{MSE}(\mathrm{track})\,/\,\Sigma w$',
            fontsize=11.5, va='center', color='#111111')
    ax.text(0.36, 0.115,
            'w = movement x validity x obj_weight,   prediction_type = sample.\n'
            'lambda weights the auxiliary track loss; lambda = 0 gives the\n'
            'action-only variant with the 391 scene tokens retained.',
            fontsize=9.6, va='center', color=P['gray_mid'], linespacing=1.7)


def panel_c(ax):
    panel(ax, 'c', 'Causal support')
    y = 0.70
    ax.annotate('', xy=(0.80, y), xytext=(0.02, y),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#333333'))
    for i, s in enumerate([0.05, 0.145, 0.24, 0.335, 0.43]):
        sel = i in (0, 2, 4)
        ax.add_patch(FancyBboxPatch(
            (s, y + 0.035), 0.082, 0.075,
            boxstyle='round,pad=0,rounding_size=0.006',
            fc=P['green_2'] if sel else 'white',
            ec=GREEN_DK if sel else P['neutral'],
            lw=1.8 if sel else 1.1,
            ls='-' if sel else (0, (3, 2)), zorder=3))
    ax.plot([0.545, 0.545], [y - 0.035, y + 0.21], color='#111111', lw=2.2,
            zorder=4)
    ax.add_patch(FancyBboxPatch((0.575, y + 0.035), 0.10, 0.075,
                                boxstyle='round,pad=0,rounding_size=0.006',
                                fc=P['red_1'], ec=P['red_strong'], lw=1.8,
                                zorder=3))
    ax.text(0.24, y + 0.235, 'support  ·  wholly elapsed,  s + H <= a',
            fontsize=10, color=GREEN_DK, ha='center', fontweight='bold')
    ax.text(0.545, y + 0.225, 'anchor  a', fontsize=10, ha='center',
            fontweight='bold', color='#111111')
    ax.text(0.625, y + 0.135, 'query', fontsize=10, color=P['red_strong'],
            ha='center', fontweight='bold')
    ax.text(0.24, y - 0.050, 'action executed, fed clean',
            fontsize=9.6, color=GREEN_DK, ha='center', va='top')
    ax.text(0.655, y - 0.050, 'action is the target,\nnoised at t ~ U',
            fontsize=9.6, color=P['red_strong'], ha='center', va='top',
            linespacing=1.6)
    arrow(ax, (0.68, y + 0.072), (0.95, y + 0.072), P['red_strong'], 1.8,
          rad=-0.18)
    ax.text(0.83, y + 0.205, 'the frame the policy\nmust act on',
            fontsize=9.4, color=P['red_strong'], ha='center', va='center',
            style='italic', linespacing=1.6)

    ax.text(0.02, 0.325,
            'At the anchor a, m windows are kept from the elapsed prefix and spread\n'
            'evenly over it, so the support is determined by (episode, a) alone and is\n'
            'independent of how long the episode turns out to be.',
            fontsize=9.8, va='top', color='#333333', linespacing=1.75)
    ax.text(0.02, 0.095,
            'The anchor quantises the query to a refit every N frames, so the cadence is\n'
            'part of the learned procedure rather than a test-time approximation of it.',
            fontsize=9.8, va='top', color='#333333', linespacing=1.75)


def panel_d(ax):
    panel(ax, 'd', 'Meta-training step')

    box(ax, 0.0, 0.615, 0.10, 0.170, 'q0', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    ax.text(0.05, 0.800, 'learned prior', fontsize=9.2, ha='center',
            va='bottom', color=P['gray_mid'])

    ax.add_patch(FancyBboxPatch((0.145, 0.465), 0.33, 0.465,
                                boxstyle='round,pad=0,rounding_size=0.02',
                                fc=P['green_1'], ec=GREEN_DK, lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.16, 0.890, 'inner  ·  create_graph=True', fontsize=9.6,
            color=GREEN_DK, va='center', fontweight='bold')
    box(ax, 0.175, 0.710, 0.27, 0.125, 'track model', 'on m past windows',
        fc=P['blue_main'], ec=P['blue_main'], fs=10)
    box(ax, 0.175, 0.528, 0.27, 0.125, 'object-only L2', 'K steps on q',
        fc='white', ec=GREEN_DK, fs=10)
    arrow(ax, (0.31, 0.707), (0.31, 0.658), GREEN_DK, 1.7)

    box(ax, 0.520, 0.615, 0.10, 0.170, 'qK', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    arrow(ax, (0.103, 0.698), (0.172, 0.750), GREEN_DK, 2.1)
    arrow(ax, (0.478, 0.698), (0.517, 0.698), GREEN_DK, 2.1)

    ax.add_patch(FancyBboxPatch((0.665, 0.465), 0.335, 0.465,
                                boxstyle='round,pad=0,rounding_size=0.02',
                                fc=P['red_1'], ec=P['red_strong'], lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.680, 0.890, 'outer  ·  all parameters', fontsize=9.6,
            color=P['red_strong'], va='center', fontweight='bold')
    box(ax, 0.695, 0.710, 0.275, 0.125, 'joint DiT', 'query frame, both noised',
        fc=P['blue_secondary'], ec=P['blue_main'], fs=10)
    box(ax, 0.695, 0.528, 0.275, 0.125, 'L_out', 'MSE(a)  +  lambda · track',
        fc='white', ec=P['red_strong'], fs=10)
    arrow(ax, (0.8325, 0.707), (0.8325, 0.658), P['red_strong'], 1.7)
    arrow(ax, (0.623, 0.698), (0.692, 0.750), GREEN_DK, 2.1)

    arrow(ax, (0.70, 0.522), (0.055, 0.610), P['red_strong'], 2.0,
          ls=(0, (6, 3)), rad=-0.16, z=5)
    ax.text(0.50, 0.360,
            'the outer gradient is taken through the inner update, into q0, alpha,\n'
            'the track model and the DiT',
            fontsize=9.8, ha='center', va='center', color=P['red_strong'],
            style='italic', linespacing=1.75)

    ax.text(0.0, 0.250,
            'There is no latent bank: every episode starts at q0 and its belief exists only\n'
            'for the duration of one forward pass, so meta-training and test-time\n'
            'adaptation run the identical procedure.',
            fontsize=9.8, va='top', color='#333333', linespacing=1.75)
    ax.text(0.0, 0.035,
            'History dropout skips the inner loop on a fraction of episodes, keeping q0 a\n'
            'usable blind prior.',
            fontsize=9.8, va='top', color='#333333', linespacing=1.75)


def panel_e(ax):
    panel(ax, 'e', 'Test-time rollout')

    steps = [('observe', 'frame t', 'white', P['gray_mid']),
             ('refit q', 'K steps, elapsed only', P['green_2'], GREEN_DK),
             ('sample', 'DDIM, action + track', P['red_1'], P['red_strong']),
             ('execute', 'n_action_steps', 'white', P['gray_mid'])]
    for i, (lab, sub, fc, ec) in enumerate(steps):
        x = i * 0.135
        box(ax, x, 0.545, 0.118, 0.330, lab, sub, fc=fc, ec=ec, fs=10)
        if i < 3:
            arrow(ax, (x + 0.120, 0.710), (x + 0.133, 0.710), P['gray_mid'],
                  1.7)
    arrow(ax, (0.462, 0.540), (0.059, 0.540), P['gray_mid'], 1.6, rad=-0.30)
    ax.text(0.26, 0.300, 'repeat each chunk  ·  refit every N frames',
            fontsize=9.6, ha='center', color=P['gray_mid'], style='italic')

    ax.text(0.56, 0.905, 'What the pipeline is scored on',
            fontsize=11, fontweight='bold', va='top', color='#111111')
    ax.text(0.56, 0.780,
            'Action   rollout success in simulation, and chunk MSE.\n\n'
            'Track, teacher-forced   the action is pinned clean and the track denoised\n'
            '    alone, which is the setting comparable to an action-conditioned\n'
            '    track model.\n\n'
            'Track, free-running   the action is sampled, so the track is predicted\n'
            '    from the policy\'s own intent.',
            fontsize=9.8, va='top', color='#333333', linespacing=1.8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/user/DyWA/output/mass'
    fig = plt.figure(figsize=(17.5, 14.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.90, 0.46],
                          hspace=0.36, wspace=0.17,
                          left=0.035, right=0.978, top=0.855, bottom=0.030)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))
    panel_e(fig.add_subplot(gs[2, :]))

    fig.text(0.035, 0.985,
             'A meta-learned dynamics belief for a joint action + track diffusion policy',
             fontsize=20, fontweight='bold', va='top')
    fig.text(0.035, 0.955,
             'A track Transformer fits an 8-d belief from already-elapsed motion; a joint '
             'DiT conditioned on that belief denoises the action chunk and the scene tracks together.\n'
             'The two models share no weights — the belief is the only thing that passes '
             'between them, and it is trained end to end through the inner update that produces it.',
             fontsize=11.5, va='top', color='#444444', linespacing=1.6)

    os.makedirs(out, exist_ok=True)
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {}), ('svg', {})):
        q = os.path.join(out, 'fig_joint_pipeline_clean.%s' % ext)
        fig.savefig(q, bbox_inches='tight', pad_inches=0.3,
                    facecolor='white', **kw)
        print('[fig] %s' % q)


if __name__ == '__main__':
    main()
