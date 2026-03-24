
import os
import sys
import glob
import json
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import zipfile

#Trouver le type des états quantiques "normaux"
def foundation_state_get_state1(state : nk.vqs.MCState):
    model=state.model
    sampler=state.sampler
    n_samples=state.n_samples
    n_discard_per_chain=state.n_discard_per_chain
    variables=state.variables
    mc_state=nk.vqs.MCState(
        sampler=sampler,
        model=model,
        variables=variables,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain
    )

    mc_state.sampler_state = mc_state.sampler.init_state(
    mc_state.sampler,
    mc_state.model,
    mc_state.variables,
    seed=0
    )

    mc_state.sampler_state = mc_state.sampler_state.replace(
    initial_config = mc_state.sampler_state.σ   
    )
    return mc_state
#def foundation_state_get_state1(sampler, n_samples,n_discard_per_chain,state : nk.State):
    model=state.model
    variables=state.variables
    mc_state1=nk.vqs.MCState(
        sampler=sampler,
        model=model,
        variables=variables,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain
    )
    mc_state2=nk.vqs.MCState(
        sampler=sampler,
        model=model,
        variables=variables,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain
    )

    mc_state1.sampler_state = mc_state1.sampler.init_state(
    mc_state1.sampler,
    mc_state1.model,
    mc_state1.variables,
    seed=0
    )

    mc_state1.sampler_state = mc_state1.sampler_state.replace(
    initial_config = mc_state2.samples 
    #cela donne la configuration à la fin de l'échantillonnage de mc_state2 donc c'est de l'état state 
    )
    return mc_state1