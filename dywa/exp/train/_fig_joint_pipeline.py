#!/usr/bin/env python3
"""Publication figure: the joint action+track diffusion pipeline with a
meta-learned dynamics belief.

    python3 _fig_joint_pipeline.py [outdir]

Supersedes `_fig_joint_arch.py`, which drew a SINGLE shared trunk. The design
settled on two separate models bridged only by the belief, in the shape of the
pick-bar reference (`ttt_with_dyn_bar.py`), so that is what this draws.

House style (figures4papers): Arial/Helvetica fallback stack, all spines off
(these are schematics, not data plots), the blue / green / red / neutral
palette, 300 dpi PNG plus vector PDF and SVG with `svg.fonttype='none'` so the
text stays editable.

Semantic colour mapping, held across all six panels:
    blue    the models -- track Transformer (inner) and joint DiT (outer)
    green   the BELIEF path: q, z_mlp, FiLM, and the ground-truth track that
            anchors the inner loss. The object under study.
    red     gradient flow and the traps that fail silently
    neutral observations, tokens with no modality
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
     'neutral': '#CFCECE', 'gray_mid': '#767676', 'gray_dark': '#4D4D4D',
     'highlight': '#FFD700'}
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
    ax.text(0.0, 1.035, tag, fontsize=15, fontweight='bold', va='bottom')
    ax.text(0.035, 1.04, title, fontsize=12, va='bottom', color='#333333')


# ---------------------------------------------------------------------------
def panel_a(ax):
    """The inner model: a belief fitted from track-prediction error."""
    panel(ax, 'a', 'INNER  —  fit the belief from elapsed motion')

    ax.text(0.0, 0.955,
            'per support window:  DINOv2 391x64  ·  patch centres 391x3  ·  EEF pose 7\n'
            'plus the EXECUTED action chunk 16x20  —  a fact, not a prediction',
            fontsize=9.3, va='top', color=P['gray_mid'], linespacing=1.7)

    box(ax, 0.03, 0.50, 0.42, 0.175, 'Track Transformer',
        'predicts 391 x 16 x 3', fc=P['blue_main'], ec=P['blue_main'], fs=11.5)
    box(ax, 0.54, 0.50, 0.42, 0.175, 'GT 3D track', 'measured, not learned',
        fc=P['green_2'], ec=GREEN_DK, fs=11.5)

    box(ax, 0.23, 0.265, 0.52, 0.155, 'object-only L2',
        'moving object patches, in mm', fc=P['green_1'], ec=GREEN_DK, fs=11)
    arrow(ax, (0.24, 0.50), (0.40, 0.425), P['blue_main'], 1.9)
    arrow(ax, (0.75, 0.50), (0.59, 0.425), GREEN_DK, 1.9)

    box(ax, 0.03, 0.055, 0.145, 0.115, 'q', 'belief, 8-d',
        fc=GREEN_DK, ec=GREEN_DK, fs=13)
    arrow(ax, (0.23, 0.30), (0.178, 0.145), P['red_strong'], 1.9, rad=0.22)
    arrow(ax, (0.103, 0.172), (0.12, 0.498), GREEN_DK, 2.1)
    ax.text(0.133, 0.355, 'FiLM', fontsize=9.6, color=GREEN_DK,
            style='italic', ha='left', va='center')
    ax.text(0.235, 0.232,
            'FiLM reads q RAW — cond_dim = z_dim, no z_mlp, as the bar does',
            fontsize=9.2, color=GREEN_DK, va='center')

    ax.text(0.275, 0.165,
            r'$q \;\leftarrow\; q \;-\; \alpha \cdot g/\|g\|$'
            '          g = dL/dq,  K steps',
            fontsize=11.5, va='center', color='#111111')
    ax.text(0.275, 0.055,
            'measured |dL/dq| ~ 1.7e-5, so the bar\'s |g| > 1 cap NEVER fires\n'
            'and q_k == q0.  Normalise always; learn alpha.',
            fontsize=9.4, va='bottom', color=P['red_strong'], linespacing=1.7)


    ax.text(0.62, 0.90,
            'the SAME q conditions\nthe DiT in (b) — that\n'
            'bridge is the only thing\ncrossing between them',
            fontsize=9.4, va='top', color=GREEN_DK, linespacing=1.7,
            style='italic')


def panel_b(ax):
    """The outer model: GAP's joint denoiser."""
    panel(ax, 'b', 'OUTER  —  the GAPPolicy joint DiT')

    ax.text(0.0, 0.975, 'context encoder carries NO action tokens  (n_prefix = 2)',
            fontsize=9.4, va='top', color=P['red_strong'])

    box(ax, 0.0, 0.645, 0.30, 0.205, 'Context encoder',
        '[CLS, state, 391 patches]', fc=P['blue_main'], ec=P['blue_main'],
        fs=10.8)
    box(ax, 0.0, 0.385, 0.30, 0.115, 'FiLM on every FFN',
        'cond_dim = z_dim, no z_mlp', fc=P['green_2'], ec=GREEN_DK, fs=10.5)
    box(ax, 0.0, 0.195, 0.125, 0.105, 'q', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    arrow(ax, (0.0625, 0.302), (0.0625, 0.382), GREEN_DK, 2.0)
    arrow(ax, (0.15, 0.503), (0.15, 0.642), GREEN_DK, 2.0)
    arrow(ax, (0.302, 0.443), (0.358, 0.470), GREEN_DK, 2.0)
    arrow(ax, (0.302, 0.748), (0.358, 0.720), P['blue_main'], 2.2)

    ax.add_patch(FancyBboxPatch((0.36, 0.29), 0.40, 0.60,
                                boxstyle='round,pad=0,rounding_size=0.018',
                                fc=P['blue_pale'], ec=P['blue_main'], lw=1.8,
                                zorder=1))
    ax.text(0.56, 0.855, 'Joint DiT  ·  self-attention only',
            fontsize=10.6, ha='center', va='center',
            fontweight='bold', color=P['blue_main'], zorder=4)

    toks = [(0.730, '16  noisy ACTION tokens', P['red_1'], P['red_strong']),
            (0.625, '391  fused TRACK tokens', '#D6E3F2', P['blue_secondary']),
            (0.520, '1  state token', 'white', P['gray_mid']),
            (0.415, '1  diffusion-timestep token', 'white', P['gray_mid'])]
    for y, lab, fc, ec in toks:
        box(ax, 0.378, y, 0.364, 0.082, lab, None, fc=fc, ec=ec, lw=1.5,
            fs=9.6)
    ax.text(0.56, 0.350, '409 tokens  ·  ONE shared timestep',
            fontsize=9.0, ha='center', va='center', color=P['blue_main'],
            style='italic', zorder=4)

    box(ax, 0.785, 0.695, 0.175, 0.105, 'action head', '16 x 20',
        fc=P['red_1'], ec=P['red_strong'], fs=9.6)
    box(ax, 0.785, 0.545, 0.175, 0.105, 'track head', '391 x 16 x 3',
        fc='#D6E3F2', ec=P['blue_secondary'], fs=9.6)
    arrow(ax, (0.762, 0.740), (0.781, 0.748), P['red_strong'], 1.9)
    arrow(ax, (0.762, 0.630), (0.781, 0.598), P['blue_secondary'], 1.9)
    # the bar gives q three routes, not one: FiLM, concat-q at the heads,
    # and a zero-init residual straight onto the output
    ax.plot([0.982, 0.982], [0.588, 0.752], color=GREEN_DK, lw=1.8, zorder=4)
    arrow(ax, (0.982, 0.748), (0.968, 0.748), GREEN_DK, 1.6, head=8)
    arrow(ax, (0.982, 0.598), (0.968, 0.598), GREEN_DK, 1.6, head=8)
    arrow(ax, (0.982, 0.500), (0.982, 0.582), GREEN_DK, 1.6, head=8)
    ax.text(0.785, 0.480,
            'q also reaches the\nheads DIRECTLY:\nconcat-q plus a\nzero-init residual',
            fontsize=8.8, va='top', color=GREEN_DK, linespacing=1.6,
            style='italic')

    ax.text(0.36, 0.235,
            r'$L_{\mathrm{out}} = \mathrm{MSE}(a) \;+\; \lambda\,'
            r'\Sigma w\,\mathrm{MSE}(\mathrm{track})/\Sigma w$'
            '       w = movement x valid x obj_weight 9',
            fontsize=10.2, va='center', color='#111111')
    ax.text(0.36, 0.145,
            'OPTION  lambda = 0 : keep the 391 tokens, drop the track head —\n'
            'same compute and parameters, so it isolates the supervision.',
            fontsize=9.4, va='center', color=P['blue_main'], linespacing=1.7)
    ax.text(0.36, 0.035,
            'Deleting the TOKENS instead blinds the policy: in GAP they ARE\n'
            'the scene conditioning, and there is no cross-attention.',
            fontsize=9.4, va='center', color=P['red_strong'], linespacing=1.7)


def panel_c(ax):
    """Which windows the belief may be fitted on."""
    panel(ax, 'c', 'Causal support  —  and the one moment that plays two roles')
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
    ax.plot([0.545, 0.545], [y - 0.09, y + 0.21], color='#111111', lw=2.2,
            zorder=4)
    ax.add_patch(FancyBboxPatch((0.575, y + 0.035), 0.10, 0.075,
                                boxstyle='round,pad=0,rounding_size=0.006',
                                fc=P['red_1'], ec=P['red_strong'], lw=1.8,
                                zorder=3))

    ax.text(0.24, y + 0.235, 'SUPPORT  ·  wholly elapsed,  s + H <= a',
            fontsize=9.8, color=GREEN_DK, ha='center', fontweight='bold')
    ax.text(0.545, y + 0.225, 'anchor  a', fontsize=9.8, ha='center',
            fontweight='bold', color='#111111')
    ax.text(0.625, y + 0.135, 'QUERY', fontsize=9.8, color=P['red_strong'],
            ha='center', fontweight='bold')
    ax.text(0.24, y - 0.025, 'action = EXECUTED, fed clean',
            fontsize=9.4, color=GREEN_DK, ha='center', va='top')
    ax.text(0.625, y - 0.025, 'action = TARGET,\nnoised at t ~ U',
            fontsize=9.4, color=P['red_strong'], ha='center', va='top',
            linespacing=1.6)

    arrow(ax, (0.68, y + 0.072), (0.96, y + 0.072), P['red_strong'], 1.8,
          rad=-0.18)
    ax.text(0.83, y + 0.20, 'the same moment\nthe policy must act on',
            fontsize=9.2, color=P['red_strong'], ha='center', va='center',
            style='italic', linespacing=1.6)

    ax.text(0.02, 0.33,
            'm = 8 windows kept of the ~76 elapsed, spread evenly.  Raising m drives\n'
            'cos(g_m, g_all) from 0.24 to 1.00, yet post error is FLAT from m = 4 up —\n'
            'so the cap is not a corner cut.',
            fontsize=9.6, va='top', color='#333333', linespacing=1.7)
    ax.text(0.02, 0.075,
            'Support and query come from ONE trajectory 16 frames apart, so the belief\n'
            'can encode pose rather than physics.  Run the same-mass swap control.',
            fontsize=9.6, va='top', color=P['red_strong'], linespacing=1.7)


def panel_d(ax):
    """The meta-step."""
    panel(ax, 'd', 'Meta-training step  —  second order, no latent bank')

    box(ax, 0.0, 0.605, 0.10, 0.165, 'q0', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    ax.text(0.05, 0.785, 'trained prior', fontsize=9.0, ha='center',
            va='bottom', color=P['gray_mid'])

    ax.add_patch(FancyBboxPatch((0.145, 0.46), 0.33, 0.455,
                                boxstyle='round,pad=0,rounding_size=0.02',
                                fc=P['green_1'], ec=GREEN_DK, lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.16, 0.875, 'INNER  ·  create_graph=True', fontsize=9.4,
            color=GREEN_DK, va='center', fontweight='bold')
    box(ax, 0.175, 0.700, 0.27, 0.125, 'track model', 'on m past windows',
        fc=P['blue_main'], ec=P['blue_main'], fs=10)
    box(ax, 0.175, 0.520, 0.27, 0.125, 'object-only L2', 'K gradient steps on q',
        fc='white', ec=GREEN_DK, fs=10)
    arrow(ax, (0.31, 0.697), (0.31, 0.650), GREEN_DK, 1.7)

    box(ax, 0.520, 0.605, 0.10, 0.165, 'qK', None, fc=GREEN_DK, ec=GREEN_DK,
        fs=12)
    arrow(ax, (0.103, 0.688), (0.172, 0.740), GREEN_DK, 2.1)
    arrow(ax, (0.478, 0.688), (0.517, 0.688), GREEN_DK, 2.1)

    ax.add_patch(FancyBboxPatch((0.665, 0.46), 0.335, 0.455,
                                boxstyle='round,pad=0,rounding_size=0.02',
                                fc=P['red_1'], ec=P['red_strong'], lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.680, 0.875, 'OUTER  ·  all parameters', fontsize=9.4,
            color=P['red_strong'], va='center', fontweight='bold')
    box(ax, 0.695, 0.700, 0.275, 0.125, 'joint DiT', 'query frame, both noised',
        fc=P['blue_secondary'], ec=P['blue_main'], fs=10)
    box(ax, 0.695, 0.520, 0.275, 0.125, 'L_out', 'MSE(a)  +  lambda · track',
        fc='white', ec=P['red_strong'], fs=10)
    arrow(ax, (0.8325, 0.697), (0.8325, 0.650), P['red_strong'], 1.7)
    arrow(ax, (0.623, 0.688), (0.692, 0.740), GREEN_DK, 2.1)

    arrow(ax, (0.70, 0.515), (0.055, 0.600), P['red_strong'], 2.0,
          ls=(0, (6, 3)), rad=-0.16, z=5)
    ax.text(0.50, 0.345,
            'the outer gradient flows THROUGH the inner update into q0, log alpha,\n'
            'the track model AND the DiT  (detach_q = False)',
            fontsize=9.6, ha='center', va='center', color=P['red_strong'],
            style='italic', linespacing=1.7)

    ax.text(0.0, 0.235,
            'History dropout 0.3 skips the inner loop on 30% of episodes, so q0 stays a\n'
            'usable blind prior.  Every episode restarts from q0 — there is no bank, so\n'
            'meta-training and test-time TTT run the identical procedure.',
            fontsize=9.6, va='top', color='#333333', linespacing=1.7)
    ax.text(0.0, 0.035,
            'The outer loss is now stochastic in (t, eps):  pin noise_seed for any pre/post.',
            fontsize=9.6, va='top', color=P['red_strong'], linespacing=1.7)


def panel_e(ax):
    """Correspondence with the pick-bar reference."""
    panel(ax, 'e', 'Same paradigm as ttt_with_dyn_bar.py, one substitution')

    rows = [('hist attempt (pcd_before, action)', 'support window (obs, EXECUTED action)'),
            ('dyn(q, action, pcd_before) -> z_hat', 'track_model(q, action_seq, obs)'),
            ('outcome_enc(pcd_after) -> z', 'GT 3D track   MEASURED, not learned'),
            ('latent MSE', 'object-only L2, in metres'),
            ('gmm(q_k, eval_pcd)', 'DiT(q_k, obs_t)'),
            ('GMM NLL vs GT grasp', 'diffusion MSE vs GT action chunk'),
            ('FiLM(cond_dim = latent_dim) + concat-q',
             'FiLM(cond_dim = z_dim) + concat-q')]
    y0, dy = 0.860, 0.076
    ax.text(0.0, y0 + 0.075, 'pick-bar reference', fontsize=10,
            fontweight='bold', color=P['gray_dark'], va='center')
    ax.text(0.50, y0 + 0.075, 'this pipeline', fontsize=10,
            fontweight='bold', color=P['blue_main'], va='center')
    ax.plot([0.0, 1.0], [y0 + 0.032, y0 + 0.032], color=P['neutral'], lw=1.4)
    ax.plot([0.478, 0.478], [y0 - 6.6 * dy, y0 + 0.095],
            color=P['neutral'], lw=1.2)
    for i, (l, r) in enumerate(rows):
        y = y0 - i * dy
        ax.text(0.0, y, l, fontsize=9.3, va='center', color=P['gray_dark'])
        hi = i in (2, 6)
        ax.text(0.50, y, r, fontsize=9.3, va='center',
                color=GREEN_DK if hi else P['blue_main'],
                fontweight='bold' if hi else 'normal')

    ax.text(0.0, 0.295,
            'The learned target is gone: no OutcomeEncoder, no collapse mode, no anti-collapse\n'
            'LayerNorm.  MAML now reshapes only the predictor — less expressive, less degenerate.',
            fontsize=9.2, va='top', color=GREEN_DK, linespacing=1.7)
    ax.text(0.0, 0.190,
            'Conditioning follows it too: FiLM reads the RAW latent.  Measured, that costs\n'
            'nothing — d(gamma)/dq within 13% of the z_mlp route, at 55k params, not 3.42M.',
            fontsize=9.2, va='top', color='#111111', linespacing=1.7)


    ax.text(0.0, 0.085,
            'But the bar runs a 64-d latent into d_model 128; this is 8 into 512, sixty-four\n'
            'times narrower in relative terms.  Choose z_dim — do not inherit it.',
            fontsize=9.2, va='top', color=P['red_strong'], linespacing=1.7)


def panel_f(ax):
    """Test-time rollout and what gets measured."""
    panel(ax, 'f', 'Test-time rollout, and what to read')

    steps = [('observe', 'frame t', 'white', P['gray_mid']),
             ('refit q', 'K steps, elapsed only', P['green_2'], GREEN_DK),
             ('DDIM sample', 'action + track', P['red_1'], P['red_strong']),
             ('execute', 'n_action_steps', 'white', P['gray_mid'])]
    for i, (lab, sub, fc, ec) in enumerate(steps):
        x = i * 0.258
        box(ax, x, 0.705, 0.225, 0.195, lab, sub, fc=fc, ec=ec, fs=10)
        if i < 3:
            arrow(ax, (x + 0.227, 0.802), (x + 0.256, 0.802), P['gray_mid'],
                  1.7)
    arrow(ax, (0.886, 0.702), (0.112, 0.702), P['gray_mid'], 1.6, rad=-0.14)
    ax.text(0.50, 0.495, 'repeat each chunk  ·  refit every N = 16 frames',
            fontsize=9.3, ha='center', color=P['gray_mid'], style='italic')

    ax.text(0.0, 0.455, 'Two axes, and neither substitutes for the other',
            fontsize=10, fontweight='bold', va='top', color='#111111')
    ax.text(0.0, 0.385,
            'ACTION   rollout success in Isaac Gym plus chunk MSE.  The only readout\n'
            '    that says whether the policy works.\n'
            'TRACK, teacher-forced   action pinned clean, track denoised alone.  The ONLY\n'
            '    setting comparable to the 10.42 blind / 11.18 film line, because those\n'
            '    were measured with the true future action as an INPUT.\n'
            'TRACK, free-running   action sampled.  Honest, and comparable to nothing yet.',
            fontsize=9.4, va='top', color='#333333', linespacing=1.75)
    ax.text(0.0, 0.055,
            'Read mv:object, never the aggregate: the track loss is 87% arm / 13% object,\n'
            'and the arm is the action\'s own consequence.',
            fontsize=9.4, va='top', color=P['red_strong'], linespacing=1.7)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/user/DyWA/output/mass'
    fig = plt.figure(figsize=(17.5, 16.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.86, 0.86],
                          hspace=0.34, wspace=0.17,
                          left=0.035, right=0.978, top=0.875, bottom=0.025)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))
    panel_c(fig.add_subplot(gs[1, 0]))
    panel_d(fig.add_subplot(gs[1, 1]))
    panel_e(fig.add_subplot(gs[2, 0]))
    panel_f(fig.add_subplot(gs[2, 1]))

    fig.text(0.035, 0.988,
             'A meta-learned dynamics belief for a joint action + track diffusion policy',
             fontsize=20, fontweight='bold', va='top')
    fig.text(0.035, 0.957,
             'Two models bridged only by an 8-d belief: a track Transformer fits it from '
             'already-elapsed motion, a GAPPolicy DiT consumes it to denoise the action '
             'chunk and the scene tracks together.',
             fontsize=11.5, va='top', color='#444444')
    fig.text(0.035, 0.932,
             'Why a policy is where the belief should pay: with the REALIZED path explicit '
             'mass buys -0.04 mm (nothing); with COMMANDED actions it buys 2.7% at 2K '
             'episodes and 6.4% at 12K — and a policy is permanently in the commanded regime.',
             fontsize=11.5, va='top', color=P['red_strong'])

    os.makedirs(out, exist_ok=True)
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {}), ('svg', {})):
        q = os.path.join(out, 'fig_joint_pipeline.%s' % ext)
        fig.savefig(q, bbox_inches='tight', pad_inches=0.3,
                    facecolor='white', **kw)
        print('[fig] %s' % q)


if __name__ == '__main__':
    main()
