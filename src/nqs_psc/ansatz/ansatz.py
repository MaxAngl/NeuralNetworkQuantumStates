# Import necessary libraries
import platform
import netket as nk
import numpy as np
from functools import partial

# jax and jax.numpy
import jax
import jax.numpy as jnp

# Flax for neural network models
import flax.linen as nn

from netket.utils.types import DType, Array
from netket.graph import Lattice

from typing import Sequence


# Mean Field Ansatz
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)
        p = nn.log_sigmoid(lam * x)
        return 0.5 * jnp.sum(p, axis=-1)


# Jastrow Ansatz
class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        J = self.param("J", nn.initializers.normal(), (n_sites, n_sites), float)
        dtype = jax.numpy.promote_types(J.dtype, x.dtype)
        J = J.astype(dtype)
        x = x.astype(dtype)
        J_symm = J.T + J
        return jnp.einsum("...i,ij,...j", x, J_symm, x)


class BM(nn.Module):
    alpha: float

    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        n_hidden = int(self.alpha * n_sites)

        W = self.param(
            "W", nn.initializers.normal(), (n_sites, n_hidden), jnp.complex128
        )
        b = self.param("b", nn.initializers.normal(), (n_hidden,), jnp.complex128)

        return jnp.sum(jnp.log(jnp.cosh(jnp.matmul(x, W) + b)), axis=-1)


@jax.jit
def logcosh_expanded(z: Array) -> Array:
    return 1 / 2 * z**2 + (-1 / 12) * z**4 + (1 / 45) * z**6


@jax.jit
def logcosh_expanded_dv(z: Array) -> Array:
    return z + (-1 / 3) * z**3 + (2 / 15) * z**5


class CNN(nn.Module):
    lattice: Lattice
    kernel_size: Sequence
    channels: tuple
    param_dtype: DType = complex

    def __post_init__(self):
        self.kernel_size = tuple(self.kernel_size)
        self.channels = tuple(self.channels)
        super().__post_init__()

    def setup(self):
        if isinstance(self.kernel_size[0], int):
            self.kernels = (self.kernel_size,) * len(self.channels)
        else:
            assert len(self.kernel_size) == len(self.channels)
            self.kernels = self.kernel_size

    @nn.compact
    def __call__(self, x):
        lattice_shape = tuple(self.lattice.extent)

        x = x / np.sqrt(2)
        _, ns = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, *lattice_shape, 1)
        for i, (c, k) in enumerate(zip(self.channels, self.kernels)):
            x = nn.Conv(
                features=c,
                kernel_size=k,
                padding="SAME",
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_normal() * jnp.exp(-jnp.sqrt((jnp.arange(self.kernel_size[0])[:, None] - center_kernel)**2 + (jnp.arange(self.kernel_size[1]) - center_kernel)**2)), 
                use_bias=True,
            )(x)

            if i:
                x = logcosh_expanded_dv(x)
            else:
                x = logcosh_expanded(x)

        x = jnp.sum(x, axis=range(1, len(lattice_shape) + 1)) / np.sqrt(ns)
        x = nn.Dense(
            features=x.shape[-1], param_dtype=self.param_dtype, use_bias=False
        )(x)
        x = jnp.sum(x, axis=-1) / np.sqrt(x.shape[-1])
        return x

class CNN_exp(nn.Module):
    lattice: Lattice
    kernel_size: Sequence
    channels: tuple
    param_dtype: DType = complex

    def __post_init__(self):
        self.kernel_size = tuple(self.kernel_size)
        self.channels = tuple(self.channels)
        super().__post_init__()

    def setup(self):
        if isinstance(self.kernel_size[0], int):
            self.kernels = (self.kernel_size,) * len(self.channels)
        else:
            assert len(self.kernel_size) == len(self.channels)
            self.kernels = self.kernel_size

    @nn.compact
    def __call__(self, x):
        lattice_shape = tuple(self.lattice.extent)

        x = x / np.sqrt(2)
        _, ns = x.shape[:-1], x.shape[-1]
        x = x.reshape(-1, *lattice_shape, 1)
        for i, (c, k) in enumerate(zip(self.channels, self.kernels)):
            center_kernel = (self.kernel_size-1) // 2
            x = nn.Conv(
                features=c,
                kernel_size=k,
                padding="SAME",
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.xavier_normal()*jnp.exp(-jnp.sqrt((center_kernel**2-k**2))),
                use_bias=True,
            )(x)

            if i:
                x = logcosh_expanded_dv(x)
            else:
                x = logcosh_expanded(x)
                
        

        x = jnp.sum(x, axis=range(1, len(lattice_shape) + 1)) / np.sqrt(ns)
        x = nn.Dense(
            features=x.shape[-1], param_dtype=self.param_dtype, use_bias=False
        )(x)
        x = jnp.sum(x, axis=-1) / np.sqrt(x.shape[-1])
        return x