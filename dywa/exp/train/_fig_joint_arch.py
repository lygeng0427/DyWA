#!/usr/bin/env python3
"""Publication figure: GAPPolicy-style joint action+track diffusion with a
meta-learned latent belief in the inner loop.

    python3 _fig_joint_arch.py [outdir]

Companion to `_fig_meta_arch.py` and deliberately in the same house style, so
the two read as a series. Same semantic colour mapping, plus one addition --
GOLD is now the ACTION modality, because the whole design turns on the action
changing role: in the track model it was a conditioning INPUT, here it is a
denoising TARGET.

    blue    the shared trunk (context encoder + joint DiT)
    green   the LATENT path -- q, z_mlp, FiLM. The object under study.
    gold    the ACTION modality (noised, denoised, executed)
    red     gradient flow. Solid = inner (w.r.t. q), dashed = outer (w.r.t. theta)
    neutral observations and data
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

P = {'blue_main': '#0F4D92', 'blue_secondary': '#3775BA', 'blue_pale': '#E4EDF7',
     'green_1': '#DDF3DE', 'green_2': '#AADCA9', 'green_3': '#8BCF8B',
     'red_1': '#F6CFCB', 'red_2': '#E9A6A1', 'red_strong': '#B64342',
     'neutral': '#CFCECE', 'gold': '#E8A33D', 'gold_pale': '#FBEBD2',
     'gold_dk': '#8A6D00'}
GREEN_DK = '#2E7D46'

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'svg.fonttype': 'none',
})


def box(ax, x, y, w, h, label, sub=None, fc='white', ec=P['blue_main'],
        lw=1.8, fs=10.5, tc=None, bold=True, r=0.010, ls='-', z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=%f' % r,
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    tc = tc or ('white' if fc in (P['blue_main'], GREEN_DK, P['gold']) else '#111111')
    ax.text(x + w / 2, y + (h * 0.62 if sub else h * 0.5), label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=tc, zorder=z + 1)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center',
                fontsize=fs - 2.3, color=tc, alpha=.88, zorder=z + 1)


def arrow(ax, p0, p1, color='#333333', lw=1.8, ls='-', rad=0.0, z=2, head=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=head, lw=lw, ls=ls,
        color=color, zorder=z,
        connectionstyle='arc3,rad=%f' % rad, shrinkA=2, shrinkB=3))


def panel(ax, tag, title):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.0, 1.02, tag, fontsize=15, fontweight='bold', va='bottom')
    ax.text(0.030, 1.025, title, fontsize=12.5, va='bottom', color='#333333')


# ---------------------------------------------------------------------------
def panel_a(ax):
    """Architecture: one action-free context encoder, one joint DiT."""
    panel(ax, 'a', 'Architecture  —  GAPPolicy’s joint denoiser, '
                   'de-bimanualised, with the belief on every FFN')

    # ---- inputs ------------------------------------------------------------
    ins = [('DINOv2 patches', '391 × 64'),
           ('patch centres', '391 × 3  (m)'),
           ('EEF pose', '7')]
    for i, (lab, sub) in enumerate(ins):
        y = 0.735 - i * 0.135
        box(ax, 0.000, y, 0.148, 0.108, lab, sub, fc='white', ec='#8A8A8A',
            lw=1.3, fs=9.2, bold=False)
        arrow(ax, (0.148, y + 0.054), (0.187, 0.600), '#8A8A8A', 1.1)
    ax.text(0.074, 0.355, 'one observation\nframe', fontsize=8.8,
            ha='center', va='top', color='#666666', style='italic',
            linespacing=1.5)

    # ---- context encoder ---------------------------------------------------
    box(ax, 0.187, 0.395, 0.168, 0.420, 'Context\nencoder',
        None, fc=P['blue_main'], ec=P['blue_main'], fs=12)
    ax.text(0.271, 0.482, '2 layers  ·  d = 512\n4-D RoPE  ·  self-attn',
            fontsize=8.8, ha='center', va='center', color='white',
            alpha=.90, linespacing=1.6, zorder=5)
    ax.text(0.271, 0.838, '[ CLS,  state,  391 patches ]', fontsize=9.0,
            ha='center', va='bottom', color=P['blue_main'], fontweight='bold')
    ax.text(0.271, 0.883, 'n_prefix = 2   ·   NO action tokens',
            fontsize=8.8, ha='center', va='bottom', color=P['red_strong'],
            style='italic')

    # ---- joint DiT ---------------------------------------------------------
    DX, DW = 0.400, 0.330
    ax.add_patch(FancyBboxPatch((DX, 0.300), DW, 0.560,
                                boxstyle='round,pad=0,rounding_size=0.012',
                                fc=P['blue_pale'], ec=P['blue_main'], lw=1.8,
                                zorder=2))
    ax.text(DX + DW / 2, 0.823, 'Joint DiT   ·   self-attention only',
            fontsize=11.2, ha='center', va='center',
            fontweight='bold', color=P['blue_main'], zorder=4)
    ax.text(DX + DW / 2, 0.787, '4 layers  ·  d = 512  ·  no cross-attention',
            fontsize=8.6, ha='center', va='center', color='#333333', zorder=4)

    toks = [('16  noisy action tokens', 'action_embed(aᵗ)  +  step pos',
             P['gold_pale'], P['gold']),
            ('391  fused track tokens',
             'fuse( ctx patch ⊕ emb(trackᵗ) )', 'white',
             P['blue_secondary']),
            ('1  state token', 'read from the context memory', 'white',
             '#8A8A8A'),
            ('1  diffusion-timestep token', 'sinusoid(t) → MLP   ·  unrotated',
             'white', '#8A8A8A')]
    for i, (lab, sub, fc, ec) in enumerate(toks):
        y = 0.675 - i * 0.092
        box(ax, DX + 0.016, y, DW - 0.032, 0.076, lab, sub, fc=fc, ec=ec,
            lw=1.5, fs=9.0, bold=True, z=4)
    ax.text(DX + DW / 2, 0.325, '409 tokens, one sequence, one shared timestep t',
            fontsize=8.9, ha='center', va='center', color=P['blue_main'],
            style='italic', zorder=4)
    arrow(ax, (0.355, 0.600), (DX, 0.600), P['blue_main'], 2.0)

    # ---- heads / outputs ---------------------------------------------------
    box(ax, 0.772, 0.660, 0.228, 0.112, 'action head',
        '→  16 × 20  commanded', fc='white', ec=P['gold'], lw=1.8,
        fs=9.8)
    box(ax, 0.772, 0.505, 0.228, 0.112, 'track head',
        '→  391 × 16 × 3  (m)', fc='white', ec=P['blue_main'],
        lw=1.8, fs=9.8)
    arrow(ax, (DX + DW, 0.713), (0.772, 0.716), P['gold'], 2.0)
    arrow(ax, (DX + DW, 0.583), (0.772, 0.561), P['blue_secondary'], 2.0)

    ax.text(0.886, 0.455,
            r'$L_{\mathrm{out}} = \mathrm{MSE}(a) + \lambda\,\Sigma w\,'
            r'\mathrm{MSE}(\mathrm{track})/\Sigma w$',
            fontsize=9.4, ha='center', va='top', color='#111111')
    ax.text(0.886, 0.405,
            'prediction_type = sample\nw = movement × valid × obj_weight 9',
            fontsize=8.5, ha='center', va='top', color='#555555',
            linespacing=1.6)

    # ---- belief path -------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0.000, 0.035), 0.730, 0.175,
                                boxstyle='round,pad=0,rounding_size=0.012',
                                fc=P['green_1'], ec=GREEN_DK, lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    box(ax, 0.022, 0.072, 0.098, 0.104, 'q', 'belief, 8-d', fc=GREEN_DK,
        ec=GREEN_DK, fs=13)
    box(ax, 0.158, 0.072, 0.140, 0.104, 'z_mlp', '8 → 512 → 512',
        fc=P['green_2'], ec=GREEN_DK, fs=10)
    box(ax, 0.336, 0.072, 0.150, 0.104, 'FiLM  γ, β',
        'per layer, pre-residual', fc=P['green_2'], ec=GREEN_DK, fs=10)
    arrow(ax, (0.120, 0.124), (0.158, 0.124), GREEN_DK, 2.2)
    arrow(ax, (0.298, 0.124), (0.336, 0.124), GREEN_DK, 2.2)
    arrow(ax, (0.395, 0.176), (0.271, 0.395), GREEN_DK, 2.2, rad=-0.12)
    arrow(ax, (0.455, 0.176), (0.520, 0.300), GREEN_DK, 2.2, rad=0.10)
    ax.text(0.600, 0.122,
            'modulates every FFN of BOTH stacks — the one path\n'
            'attention cannot route around',
            fontsize=8.9, color=GREEN_DK, style='italic', va='center',
            ha='center', linespacing=1.6)

    ax.text(0.886, 0.175,
            'Single 7-D EEF, so GAP’s per-arm token\nsplit is dropped: '
            '1 state token, 16 action\ntokens (one per step), not 2 and 2H.\n'
            'Action RoPE xyz is τ-blended from the\nNOISY chunk itself '
            '— never ground truth,\nidentical at train and test.',
            fontsize=8.6, ha='center', va='center', color='#333333',
            linespacing=1.75)


# ---------------------------------------------------------------------------
def panel_b(ax):
    """The action channel: a fact in the past, a target in the future."""
    panel(ax, 'b', 'The action channel changes role at the anchor')

    y = 0.655
    ax.annotate('', xy=(0.99, y), xytext=(0.02, y),
                arrowprops=dict(arrowstyle='-|>', lw=1.6, color='#333333'))
    A, H = 0.600, 0.105

    for i, s in enumerate([0.055, 0.155, 0.255, 0.355, 0.455]):
        sel = i in (0, 2, 4)
        ax.add_patch(Rectangle((s, y + 0.030), H, 0.070,
                               fc=P['green_2'] if sel else 'white',
                               ec=GREEN_DK if sel else '#B5B5B5',
                               lw=1.6 if sel else 1.0, zorder=3))
    ax.add_patch(Rectangle((0.650, y + 0.030), 0.175, 0.070,
                           fc=P['gold_pale'], ec=P['gold'], lw=1.8, zorder=3))

    ax.plot([A, A], [y - 0.075, y + 0.225], color='#111111', lw=2.2, zorder=4)
    ax.text(A, y + 0.248, 'anchor  a', fontsize=10, ha='center',
            fontweight='bold')
    ax.text(0.255, y + 0.148, 'SUPPORT  ·  elapsed', fontsize=9.4,
            color=GREEN_DK, ha='center', fontweight='bold')
    ax.text(0.738, y + 0.148, 'QUERY', fontsize=9.4, color=P['gold_dk'],
            ha='center', fontweight='bold')

    ax.text(0.255, y - 0.055, 'action  =  EXECUTED\nfed clean, σ = 0',
            fontsize=9.0, ha='center', va='top', color=GREEN_DK,
            linespacing=1.6)
    ax.text(0.738, y - 0.055, 'action  =  TARGET\nnoised at t ∼ U',
            fontsize=9.0, ha='center', va='top', color=P['gold_dk'],
            linespacing=1.6)

    ax.text(0.02, 0.385,
            'Support windows lie wholly in the past (s + H ≤ a), so their actions '
            'are facts. The\nquery’s action is what the policy must produce, so it '
            'may enter ONLY as noise.',
            fontsize=9.6, va='top', color='#333333', linespacing=1.65)
    ax.text(0.02, 0.150,
            'This is why the context encoder carries no action tokens — with GAP’s '
            'layout there is\nno path for the query action to leak, by construction '
            'rather than by discipline.',
            fontsize=9.6, va='top', color=P['red_strong'], linespacing=1.65,
            style='italic')


# ---------------------------------------------------------------------------
def panel_c(ax):
    """The meta-step."""
    panel(ax, 'c', 'Meta-training step')

    box(ax, 0.000, 0.620, 0.084, 0.150, 'q\u2080', 'prior', fc=GREEN_DK,
        ec=GREEN_DK, fs=12)

    ax.add_patch(FancyBboxPatch((0.106, 0.455), 0.420, 0.455,
                                boxstyle='round,pad=0,rounding_size=0.015',
                                fc=P['green_1'], ec=GREEN_DK, lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.120, 0.878, 'INNER  \u00b7  latent only  \u00b7  create_graph=True',
            fontsize=9.2, color=GREEN_DK, fontweight='bold', va='center')

    box(ax, 0.128, 0.700, 0.170, 0.112, 'support', 'm windows, past',
        fc='white', ec='#8A8A8A', lw=1.3, fs=9.4, bold=False)
    box(ax, 0.128, 0.560, 0.170, 0.112, 'joint DiT', 'pinned (t_in, seed)',
        fc=P['blue_pale'], ec=P['blue_main'], lw=1.5, fs=9.4)
    arrow(ax, (0.213, 0.700), (0.213, 0.672), '#8A8A8A', 1.4)
    box(ax, 0.338, 0.590, 0.172, 0.222,
        'object-only L2', 'moving object patches,\nraw metres',
        fc='white', ec=GREEN_DK, lw=1.6, fs=9.4)
    arrow(ax, (0.298, 0.616), (0.338, 0.660), GREEN_DK, 1.6)
    ax.text(0.316, 0.503,
            r'$q \leftarrow q - \alpha\, g/\|g\|$,     K steps',
            fontsize=10.2, ha='center', va='center', color=GREEN_DK)
    arrow(ax, (0.424, 0.590), (0.316, 0.536), P['red_strong'], 1.8, rad=0.25)

    box(ax, 0.556, 0.620, 0.084, 0.150, 'q\u2096', 'belief', fc=GREEN_DK,
        ec=GREEN_DK, fs=12)
    arrow(ax, (0.084, 0.695), (0.128, 0.740), GREEN_DK, 2.0)
    arrow(ax, (0.526, 0.695), (0.556, 0.695), GREEN_DK, 2.0)

    ax.add_patch(FancyBboxPatch((0.668, 0.455), 0.332, 0.455,
                                boxstyle='round,pad=0,rounding_size=0.015',
                                fc=P['gold_pale'], ec=P['gold'], lw=1.5,
                                ls=(0, (5, 3)), zorder=1))
    ax.text(0.682, 0.878, 'OUTER  \u00b7  all parameters', fontsize=9.2,
            color=P['gold_dk'], fontweight='bold', va='center')
    box(ax, 0.696, 0.700, 0.276, 0.112, 'query frame',
        'action + track, both noised', fc='white', ec='#8A8A8A', lw=1.3,
        fs=9.4, bold=False)
    box(ax, 0.696, 0.560, 0.276, 0.112, 'joint denoising loss',
        'MSE(a)  +  \u03bb \u00b7 track', fc='white', ec=P['gold'], lw=1.8,
        fs=9.4)
    arrow(ax, (0.834, 0.700), (0.834, 0.672), '#8A8A8A', 1.4)
    arrow(ax, (0.640, 0.695), (0.696, 0.740), GREEN_DK, 2.0)

    arrow(ax, (0.700, 0.545), (0.060, 0.605), P['red_strong'], 1.9,
          ls=(0, (6, 3)), rad=-0.16, z=5)
    ax.text(0.380, 0.368,
            'the outer gradient flows THROUGH the inner update into\n'
            'q\u2080, log \u03b1, z_mlp, the encoder, the DiT and both heads',
            fontsize=9.2, ha='center', va='center', color=P['red_strong'],
            style='italic', linespacing=1.6)

    ax.text(0.000, 0.250,
            'History dropout 0.3 skips the inner loop on 30 % of episodes, so '
            'q\u2080 stays a\nusable blind prior and pre-vs-post stays meaningful.',
            fontsize=9.4, va='top', color='#333333', linespacing=1.65)
    ax.text(0.000, 0.120,
            'OPEN PARAMETER  t_in.  The inner loss is now a denoising loss, so the\n'
            'belief\u2019s gradient dies at both ends \u2014 near t = T the prior dominates,\n'
            'near t = 0 the task is trivial.  Sweep it first; the fallback is a\n'
            'deterministic auxiliary track head as the scorer, which removes t_in.',
            fontsize=9.2, va='top', color=P['red_strong'], linespacing=1.6)


# ---------------------------------------------------------------------------
def panel_d(ax):
    panel(ax, 'd', 'Test-time rollout, and what gets measured')

    steps = [('observe', 'frame t'),
             ('refit q', 'K steps, elapsed\nwindows only'),
             ('DDIM sample', 'action + track\nin lockstep'),
             ('execute', 'n_action_steps\nof the chunk')]
    for i, (lab, sub) in enumerate(steps):
        x = 0.000 + i * 0.158
        fc = {1: P['green_2'], 2: P['gold_pale']}.get(i, 'white')
        ec = {1: GREEN_DK, 2: P['gold']}.get(i, '#8A8A8A')
        box(ax, x, 0.560, 0.136, 0.300, lab, sub, fc=fc, ec=ec, lw=1.6,
            fs=9.8)
        if i < 3:
            arrow(ax, (x + 0.136, 0.710), (x + 0.158, 0.710), '#555555', 1.6)
    # loop arrow routed ABOVE the row: below it crossed the caption text
    arrow(ax, (0.542, 0.868), (0.068, 0.868), '#555555', 1.5, rad=0.15)
    ax.text(0.305, 0.945, 'repeat each chunk', fontsize=8.9, ha='center',
            color='#555555', style='italic')
    ax.text(0.226, 0.505, 'refit every N = 16 frames', fontsize=8.9,
            ha='center', color=GREEN_DK, style='italic')

    ax.text(0.000, 0.355,
            'The belief persists across the chunk; the diffusion\n'
            'sample does not. q is refit from executed motion\n'
            'only, so the rollout is causal by exactly the rule\n'
            'meta-training used.',
            fontsize=9.4, va='top', color='#333333', linespacing=1.65)

    ax.text(0.620, 0.900, 'Two axes, and neither substitutes for the other',
            fontsize=10.2, va='top', fontweight='bold', color='#111111')
    ax.text(0.620, 0.825,
            'ACTION   rollout success in Isaac Gym + chunk MSE.  The only\n'
            '    readout that answers whether the policy works.\n\n'
            'TRACK, teacher-forced   action pinned clean at σ = 0, track\n'
            '    denoised alone.  The ONLY setting comparable to the\n'
            '    10.42 blind / 11.18 film line \u2014 those numbers were measured\n'
            '    with the true future action as an INPUT.\n\n'
            'TRACK, free-running   action sampled.  The honest joint number,\n'
            '    and NOT comparable to anything measured so far.',
            fontsize=8.9, va='top', color='#333333', linespacing=1.62)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '/home/user/DyWA/output/mass'
    fig = plt.figure(figsize=(16.0, 13.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.26, 0.80, 0.62],
                          hspace=0.30, wspace=0.14,
                          left=0.032, right=0.978, top=0.868, bottom=0.028)
    panel_a(fig.add_subplot(gs[0, :]))
    panel_b(fig.add_subplot(gs[1, 0]))
    panel_c(fig.add_subplot(gs[1, 1]))
    panel_d(fig.add_subplot(gs[2, :]))

    fig.text(0.032, 0.988, 'Joint action + track diffusion with a '
             'test-time-adapted dynamics belief',
             fontsize=19, fontweight='bold', va='top')
    fig.text(0.032, 0.959,
             'The track model is promoted to a policy. The belief that was '
             'worth little when the future action was given should matter '
             'most when it has to be chosen:',
             fontsize=11.2, va='top', color='#444444')
    fig.text(0.032, 0.933,
             'with the REALIZED path, explicit mass buys −0.04 mm (nothing); '
             'with COMMANDED actions it buys 2.7 % at 2K episodes and 6.4 % at '
             '12K — and a policy is permanently in the commanded regime.',
             fontsize=11.2, va='top', color=P['red_strong'])

    os.makedirs(out, exist_ok=True)
    for ext, kw in (('png', dict(dpi=300)), ('pdf', {}), ('svg', {})):
        q = os.path.join(out, 'fig_joint_arch.%s' % ext)
        fig.savefig(q, bbox_inches='tight', pad_inches=0.25,
                    facecolor='white', **kw)
        print('[fig] %s' % q)


if __name__ == '__main__':
    main()
