#!/usr/bin/env python3
"""Track model conditioned on a FREE latent instead of the true mass.

`FricTrackModel` conditions on a privileged scalar: `batch['phys']` (the object's
mass) goes through `phys_mlp` and drives FiLM on every encoder/decoder FFN. That
works -- 6.4% better than an unconditioned model at 12K episodes, 17.1% at
stream scale -- but it needs information no robot has.

This removes the slot. `phys_mlp` is DELETED, `batch['phys']` is never read, and
the conditioning is a free latent `q` that the caller supplies per trajectory.
`q` starts at the learned prior `q0` and is adapted by gradient descent on the
episode's own ELAPSED motion -- at meta-training and at test time alike, through
one shared function (`meta_ttt.ttt_inner_update`).

ONE METHOD CHANGES. `phys_vec` is the single integration point in the parent --
it is the only consumer of `batch['phys']` anywhere in the model, and all four
conditioning call sites funnel through it (`track_model.py:203/219/253/265`).
Overriding it alone leaves `encode_context`, both trunks, `compute_loss`,
`predict` and the loss path untouched.

`phys_cond='film'` is not incidental. It is the one mode where `n_prefix` stays
`2 + H` and `_rope_obs` falls through to the base, so there is NO prefix or RoPE
bookkeeping to get wrong -- and getting that wrong is silent in the direction
that matters: the patch-count assert catches an added token but not a shifted
coordinate offset.

THE LATENT LIVES IN THE BATCH, NOT IN THE MODEL. `batch['q']` is `[B, z_dim]`,
supplied by the caller. That keeps `phys_vec` pure (the parent recomputes it
three times per forward and its docstring explains why caching would go stale),
lets evaluation feed an externally optimised `q` with no model surgery, and makes
`th.autograd.grad(loss, q, create_graph=True)` a one-liner -- which is the whole
second-order inner loop.

`q0` IS a trained parameter. It is where every trajectory starts, and it receives
gradient two ways: directly, whenever the inner loop is skipped (the trainer's
history dropout, matching the reference's `end_dropout`), and through the inner
update's graph, since `q_k = q0 - alpha * g(q0)` depends on `q0` twice. There is
NO per-episode latent bank and no EMA -- an episode's latent exists only for the
duration of one forward pass, so meta-training and test-time TTT run the same
procedure by construction rather than by agreement.

`log_alpha` IS ALSO TRAINED, and it is not a convenience. The reference's inner
step caps the gradient only when `|g| > 1`; measured here `|dL/dq| ~ 1.7e-5`, so
that cap never fires, `q_k == q0`, the outer loss sees no difference, and there
is no meta-signal to grow the gradient -- a deadlock. `ttt_inner_update` therefore
normalises the gradient ALWAYS and multiplies by `softplus(log_alpha)`, so the
step is scale-free at init and its size is meta-learned rather than guessed.
"""
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from models.fric.track_model import FricTrackModel


class LatentTrackModel(FricTrackModel):
    """`FricTrackModel` with the mass input replaced by a free latent.

    Constructed with `phys_cond='film'` so the FiLM-enabled encoders are built,
    then `phys_mlp` is removed. `phys_lo`/`phys_hi`/`phys_log` are still accepted
    and recorded (downstream checkpoint readers index them without `.get`) but
    are no longer used for anything -- there is no scalar to map.
    """

    def __init__(self, *args, z_dim: int = 8, q0_init_std: float = 0.1,
                 alpha_init: float = 0.1, direct_film: bool = False,
                 film_init_scale: float = 1.0, **kwargs):
        kwargs['phys_cond'] = 'film'
        assert z_dim >= 1, z_dim
        # `direct_film` drops `z_mlp` and lets FiLM read `q` itself, as the
        # pick-bar reference does (`cond_dim=latent_dim`). Measured at the real
        # init, that costs nothing that matters -- d(gamma)/dq lands within 13%
        # of the embedded route -- for 62x fewer conditioning parameters. It is
        # OFF by default because `film_dim` is baked into every layer's shape,
        # so flipping it makes existing checkpoints unloadable.
        if direct_film:
            kwargs['film_dim'] = int(z_dim)
        super().__init__(*args, **kwargs)
        assert alpha_init > 0.0, alpha_init
        self.z_dim = int(z_dim)
        self.direct_film = bool(direct_film)
        # The mass -> latent map is gone. Deleting rather than leaving it unused
        # means an attempt to load a mass-conditioned checkpoint into this class
        # fails loudly on the missing keys instead of silently carrying a dead
        # module that a later reader might think is live.
        del self.phys_mlp
        # See JointPolicyDiT: the FiLM init is calibrated for a d_model-wide
        # conditioning vector, so a raw z_dim-wide one needs rescaling or the
        # belief has no leverage on the loss it is fitted against -- and a flat
        # inner surface makes the unit-normalised gradient pure noise.
        if film_init_scale != 1.0:
            with th.no_grad():
                for n_, p_ in self.named_parameters():
                    if '.film.' in n_:
                        p_.mul_(float(film_init_scale))
        self.film_init_scale = float(film_init_scale)
        self.q0 = nn.Parameter(th.randn(self.z_dim) * float(q0_init_std))
        if not direct_film:
            self.z_mlp = nn.Sequential(
                nn.Linear(self.z_dim, self.d_model), nn.SiLU(),
                nn.Linear(self.d_model, self.d_model))
        # softplus^-1(alpha_init), so `alpha()` starts exactly at `alpha_init`.
        # Parameterised through softplus rather than clamped because the meta
        # gradient can and does push it negative early, and a clamp there would
        # silently zero the gradient instead of shrinking the step.
        self.log_alpha = nn.Parameter(
            th.tensor(math.log(math.expm1(float(alpha_init)))))

    def alpha(self) -> th.Tensor:
        """Inner-loop step size, strictly positive. Differentiable."""
        return F.softplus(self.log_alpha)

    # -- conditioning --------------------------------------------------------
    def phys_vec(self, batch):
        """[B, D] from `batch['q']`, falling back to the global prior.

        A missing `q` is the prior rather than an error on purpose: the
        no-evidence regime (`t < H`, no window has elapsed) is a real and
        frequent state -- 17.3% of scored frames -- and it must be expressible
        without the caller assembling a tensor of copies. It is also how the
        trainer's history dropout is spelled.
        """
        q = batch.get('q', None)
        B = batch['patch_centers'].shape[0]
        if q is None:
            q = self.q0.unsqueeze(0).expand(B, -1)
        else:
            assert q.shape == (B, self.z_dim), (q.shape, B, self.z_dim)
        return q if self.direct_film else self.z_mlp(q)

    # -- convenience for the trainer/eval ------------------------------------
    def prior(self, n: int) -> th.Tensor:
        """[n, z_dim] copies of q0, detached -- the initial value of every
        trajectory's latent, at training and at test time alike."""
        return self.q0.detach().unsqueeze(0).expand(n, -1).clone()
