from functools import partial

import jax
import jax.numpy as jnp

from flax import linen as nn

from einops import rearrange


def roll(J, shift, axis=-1):
    return jnp.roll(J, shift, axis=axis)
#permutation circulaire sur axis


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
#les vmap font en sorte qu'en appelant roll2d avec des tableaux en argument on ait une double somme sur les elements des tableaux
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)
# le format de sortie est (batch, Ni, Nj, side*side)
# et avec le reshape on remet ca en 2 dimensions cf en dessous pour signification du -1 
#Ni : nombre de shifts horizontaux et Nj verticaux que tu veux faire
class FMHA(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )
        if self.transl_invariant:
            #On voit que c'est invariant par translation car les memes poids sont appliqués à  toutes le spermutations circulaires
            self.J = self.param(
                "J", nn.initializers.xavier_uniform(), (self.h, self.L_eff), jnp.float64
            )
            # J est donc un tableau de poids de la forme (h,L_eff) et en 2D on a souvent L_eff=side²
            if self.two_dimensional:
                sq_L_eff = int(self.L_eff**0.5)
                assert sq_L_eff * sq_L_eff == self.L_eff
                self.J = roll2d(self.J, jnp.arange(sq_L_eff), jnp.arange(sq_L_eff))
                #ici J est donc de la forme (h,sq_L_eff, sq_L_eff, self.L_eff )
                self.J = self.J.reshape(self.h, -1, self.L_eff)
                # IMPT !! RESHAPE NE MODIFIE PAS LE NB TOTAL D'ELEMENTS
                #le -1 SIGNIFIE COMPLETE POUR QUE LE NB TOT D4ELEMENTS SOIT CONSERVE
                # J de la forme (h, L_eff, L_eff) 
                #donc pour chaque filtre h on a toutes ses translations chacun étant de taille L_eff
            else:
                self.J = jax.vmap(roll, (None, 0), out_axes=1)(
                    self.J, jnp.arange(self.L_eff)
                )
                #jax.vmap(f, in_axes=(None, 0))(J, shifts)
                # cette ligne dit en gros appelle f plusieurs fois et à chaque fois avec le même J (car None) et un shift différent (car 0)
                # Ex : si shifts = jnp.array([0, 1, 2])
                #[f(J, 0),
                #f(J, 1),
                #f(J, 2)]
                # et ceci avec le J donné par self.J et le shift donné par  jnp.arange(self.L_eff)
                #format final (h, L_eff, L_eff)



        else:
            #Pas d'invariance par translation ici car le spoids diffèrent tous
            self.J = self.param(
                "J",
                nn.initializers.xavier_uniform(),
                (self.h, self.L_eff, self.L_eff),
                jnp.float64,
            )

        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        #x est de la forme (batch, L_eff,d_model)
        # d'après le code au dessus J est de la forme (h, L_eff, L_eff)
        v = self.v(x)
        v = rearrange(v, "batch L_eff (h d_eff) -> batch L_eff h d_eff", h=self.h)
        v = rearrange(v, "batch L_eff h d_eff -> batch h L_eff d_eff")
        #on switche pour le calcul matriciel car on veut faire X=J*V pour CHAQUE BATCH ET CHAQUE TETE
        x = jnp.matmul(self.J, v)
        #resultat de la forme (batch, h, L_eff,d_eff)
        x = rearrange(x, "batch h L_eff d_eff  -> batch L_eff h d_eff")
        x = rearrange(x, "batch L_eff h d_eff ->  batch L_eff (h d_eff)")
        #dernière ligne, on concatène toutes les têtes

        x = self.W(x)

        return x