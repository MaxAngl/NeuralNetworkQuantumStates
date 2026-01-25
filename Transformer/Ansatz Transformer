import jax
import jax.numpy as jnp

from flax import linen as nn
from einops import rearrange
from .attentions import FMHA


def extract_patches1d(x, b):
    return rearrange(x, "batch (L_eff b) -> batch L_eff b", b=b)


def extract_patches2d(x, b):
    batch = x.shape[0]
    L_eff = int((x.shape[1] // b**2) ** 0.5)
    x = x.reshape(batch, L_eff, b, L_eff, b)  # [L_eff, b, L_eff, b]
    x = x.transpose(0, 1, 3, 2, 4)  # [L_eff, L_eff, b, b]
    # flatten the patches
    x = x.reshape(batch, L_eff, L_eff, -1)  # [L_eff, L_eff, b*b]
    x = x.reshape(batch, L_eff * L_eff, -1)  # [L_eff*L_eff, b*b]
    return x


def log_cosh(x):
    sgn_x = -2 * jnp.signbit(x.real) + 1
    x = x * sgn_x
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


class Embed(nn.Module):
    d_model: int
    b: int
    two_dimensional: bool = False

    def setup(self):
        if self.two_dimensional:
            self.extract_patches = extract_patches2d
        else:
            self.extract_patches = extract_patches1d

        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=jnp.float64,
            dtype=jnp.float64,
        )

    def __call__(self, x):
        x = self.extract_patches(x, self.b)
        x = self.embed(x)

        return x


class EncoderBlock(nn.Module):
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            h=self.h,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )
        #shape des tokens (batch, N_tokens, d_model)

        self.layer_norm_1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        self.layer_norm_2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=jnp.float64,
                    dtype=jnp.float64,
                ),
            ]
        )
        #feed forward part of the Transformer
        #nn.gelu = Gaussian error Linear Unit, stable linéaire et meilleur alternative à RELU dans Transformers
#introduce non-linearity
# ATTENTION chaquze token est traité de manière indépendante très différent de l'attention
    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


class Encoder(nn.Module):
    num_layers: int
    d_model: int
    h: int
    L_eff: int
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                h=self.h,
                L_eff=self.L_eff,
                transl_invariant=self.transl_invariant,
                two_dimensional=self.two_dimensional,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x


class OuputHead(nn.Module):
    d_model: int
    complex: bool = False

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)

        self.norm0 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )
        self.norm1 = nn.LayerNorm(
            use_scale=True, use_bias=True, dtype=jnp.float64, param_dtype=jnp.float64
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=jnp.float64,
            dtype=jnp.float64,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x, return_z=False):

        z = self.out_layer_norm(x.sum(axis=-2))

        if return_z:
            return z

        # amp = self.norm0(self.output_layer0(z))
        amp = self.output_layer0(z)
        if self.complex:
            # sign = self.norm1(self.output_layer1(z))
            sign = self.output_layer1(z)
            out = amp + 1j * sign
        else:
            out = amp

        return jnp.sum(log_cosh(out), axis=-1)
    #on a donc un scalaire par élément du batch


class ViTFNQS(nn.Module):
    num_layers: int
    d_model: int
    #dimension vecteurs d'embeddings de clés, values...
    heads: int
    L_eff: int
    #Spécifique à VMC pas dans Transfomer normaux. 
    #C'est la longueur effective : peut correspondre à la taille du système ou tronquée (d'après Chat svt nb de degrés de liberte ou patches)

    #Utile pour attention restreinte
    b: int
    #taille d'un batch local 
    #ie un token =un batch de spins b ou (b*b en 2D)
    #Cela permet de traiter EXACTEMENT les corrélations locales et c'est ce qu'on veut car on sait que les corrélatio
    #locales sont plus importantes donc on pousse le Transformer à les traiter exactement et puis on le laisse apprendre les intercations lointaines
    #b est ainsi une longueur de corrélation locale imposée par l'ansatz
    n_coups: int
    #dimension d'un vecteur de conditionnement global 
    #c'est n coups représente les paramètres d el'Hamiltonien J, h...
    #on les ajoute à tous les token en les les broadcastant(cf code en dessous) comme ca tous les tokens les connaissent et on peut determiner famille de fonctions d'onde.
    complex: bool = False
    disorder: bool = False
    transl_invariant: bool = True
    two_dimensional: bool = False

    def setup(self):
        #setup declare au début pour les sous-modules
        if self.disorder:
            self.patches_and_embed = Embed(
                self.d_model // 2, self.b, two_dimensional=self.two_dimensional
            )
            #cela fait 1) l'extraction de batchs b ie cela regroupe l'entree de la forme [up, down, up, up...] e,n batchs de taille b
            # la dimension est donc (L/b,b)
            #puis cela applique une matrice lineaire pour avoir dimension finale se.d_model/2
            self.patches_and_embed_coup = Embed(
                self.d_model // 2, self.b, two_dimensional=self.two_dimensional
            )
        else:
            self.embed = nn.Dense(
                self.d_model,
                kernel_init=nn.initializers.xavier_uniform(),
                param_dtype=jnp.float64,
                dtype=jnp.float64,
            )
            #declaration de la fully connected layer
            #Rappel : comme c'est fully connected c'est juste une matrice linéaire W avec un vecteur b
            #Attention pas encore appliquée, juste déclarée

        self.encoder = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            h=self.heads,
            L_eff=self.L_eff,
            transl_invariant=self.transl_invariant,
            two_dimensional=self.two_dimensional,
        )
        #Encoder code directement le Transformer
        # si invariant par translation, l'implémente direct doonc nb paramètres diminue

        self.output = OuputHead(self.d_model, complex=self.complex)
        #CALCUL DE phi(sigma)
        #équivalent de l'unembedding matrix sauf que l'unembedding matrix fait passer un vecteur de d_model à d_voc
        # En VMC, au lieu de retourne run vecteur on agrège tout pour avoir phi(sigma) ou son log !
# ce Outputhead va recevoir les sorties du Encoder
    def __call__(self, x):
        n_coups = self.n_coups
        spins = x[..., :-n_coups]
        # 1 ere dimension est comme d'hab les batchs et cela prend dans la 2 eme dimension tout sauf les ncoups derniers
        coups = x[..., -n_coups:]
        #prend les n coup derniers éléments (les paramètres)

        return_z = False
        x = jnp.atleast_2d(spins)
        #force spins à être au moins 2D -> garantit qu'il y ait une dimension de batch

        if self.disorder:
            # raise NotImplementedError
            coups = jnp.broadcast_to(coups, x.shape)
            #cela duplique les coups pour qu'ils aient la meme forme que les spins ex (batch, L)
            x_spins = self.patches_and_embed(x)
            x_coups = self.patches_and_embed(coups)
            #applique les opérations qu'on a vu en haut
            x = jnp.concatenate((x_spins, x_coups), axis=-1)
            #concantène du coup la forme globale est bien de(batch, L/b, d_model) d__model
        else:
            if self.two_dimensional:
                x = extract_patches2d(x, self.b)
            else:
                x = extract_patches1d(x, self.b)
                #On transforme la configuration de spins en patches, b=taille du patch
                #en 1D si on avait (batch, L) on a mtn (batch,L/b,b)
            coups = jnp.broadcast_to(
                coups.reshape(*coups.shape[:-1], 1, -1), (*x.shape[:-1], n_coups)
            )
            #le reshape fait passer de (batch, n_coups) à (batch, 1,n_coups)
            #comme x est de la forme (batch, N_batches(ie L/b), b) après patchification vu juste au dessus
            #coups.shape est finalement de la forme (batch, N_batches, n_coups)


            # coups = jnp.repeat(coups[:, None], repeats=x.shape[1], axis=1)
            x = jnp.concatenate((x, coups), axis=-1)
            #forme (batch, N_batches, b+n_coups)
            x = self.embed(x)
            #applique la Dense dont la matrice fait passer à la bonne dimension

        x = self.encoder(x)

        out = self.output(x, return_z=return_z)

        return out