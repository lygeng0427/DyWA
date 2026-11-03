"""Friction-conditioned track models.

Wraps `models.gap` rather than forking it: the trunks, the loss and the
normalizer are shared, so an A/B between `phys_cond='none'` and
`phys_cond='both'` differs in the conditioning path and nothing else.
"""
