"""
ViTFNQS with dual embedding: separate embeddings for spins and couplings.

The first d_model/2 dimensions encode spin information only,
the last d_model/2 dimensions encode coupling information only.
Mixing happens in the Encoder (attention + FFN), not at the embedding level.

Usage:
    from ansatz_dual_embed import ViTFNQS_DualEmbed

    model = ViTFNQS_DualEmbed(
        num_layers=2, d_model=16, heads=4,
        L_eff=L, b=1, n_coups=L,
        complex=True, transl_invariant=False,
    )
"""

import jax.numpy as jnp
from flax import linen as nn

from netket_foundational._src.model.vit import (
    Embed,
    Encoder,
    OuputHead,
)


class ViTFNQS_DualEmbed(nn.Module):
    """ViTFNQS variant with separate spin and coupling embeddings.

    Each token receives:
      - Its local patch of b spins  -> embedded to d_model/2
      - Its local patch of b coups  -> embedded to d_model/2
      - Concatenated               -> d_model

    This enforces structural separation: the encoder (attention + FFN)
    is the first place where spin and coupling info interact.
    """
    num_layers: int
    d_model: int
    heads: int
    L_eff: int
    b: int
    n_coups: int
    complex: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        assert self.d_model % 2 == 0, "d_model must be even for dual embedding"

        half_d = self.d_model // 2

        self.embed_spins = Embed(
            half_d, self.b, two_dimensional=self.two_dimensional
        )
        self.embed_coups = Embed(
            half_d, self.b, two_dimensional=self.two_dimensional
        )

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )

        self.output = OuputHead(self.d_model, complex=self.complex)

    def __call__(self, x):
        n_coups = self.n_coups
        spins = x[..., :-n_coups]
        hvals = x[..., -n_coups:]

        spins = jnp.atleast_2d(spins)
        hvals = jnp.atleast_2d(hvals)

        # Broadcast hvals to match spins shape (needed if n_coups < L)
        hvals = jnp.broadcast_to(hvals, spins.shape)

        # Separate embeddings: each patch sees only its local spins/couplings
        x_spins = self.embed_spins(spins)    # (batch, L_eff, d_model/2)
        x_coups = self.embed_coups(hvals)    # (batch, L_eff, d_model/2)

        # Concatenate: first half = spins, second half = couplings
        x = jnp.concatenate((x_spins, x_coups), axis=-1)  # (batch, L_eff, d_model)

        # Encoder mixes spin/coupling info via attention + FFN
        x = self.encoder(x)

        return self.output(x)
