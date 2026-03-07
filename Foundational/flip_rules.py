import jax
import jax.numpy as jnp
from flax import serialization
from netket.sampler import rules
from netket.utils import struct

# ==========================================
# 1. DÉFINITION DE LA CLASSE (Méthode Robuste)
# ==========================================
# On utilise la méthode manuelle qui a prouvé sa stabilité
class GlobalFlipRule(rules.MetropolisRule):
    
    # Champ statique pour NetKet
    prob_global: float = struct.field(pytree_node=False)

    def __init__(self, prob_global=0.1):
        self.prob_global = prob_global

    def transition(self, sampler, machine, parameters, state, key, sigma):
        n_chains = sigma.shape[0]
        L = sigma.shape[-1]
        
        key_prob, key_site = jax.random.split(key, 2)

        # Propositions
        sigma_global = -sigma
        sites = jax.random.randint(key_site, shape=(n_chains,), minval=0, maxval=L)
        mask = jax.nn.one_hot(sites, L)
        sigma_local = sigma * (1 - 2 * mask)

        # Choix
        rand_vals = jax.random.uniform(key_prob, shape=(n_chains, 1))
        sigma_prop = jnp.where(rand_vals < self.prob_global, sigma_global, sigma_local)

        return sigma_prop.astype(sigma.dtype), None

    # Méthodes JAX Pytree
    def tree_flatten(self):
        return ((), (self.prob_global,))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(prob_global=aux_data[0])
    
    # Méthodes Hachage
    def __hash__(self):
        return hash(self.prob_global)

    def __eq__(self, other):
        if not isinstance(other, GlobalFlipRule): return False
        return self.prob_global == other.prob_global

# ==========================================
# 2. ENREGISTREMENTS (JAX, Flax, et... NQXPACK)
# ==========================================

# A. Enregistrement JAX (Indispensable pour le calcul)
try:
    jax.tree_util.register_pytree_node(
        GlobalFlipRule,
        GlobalFlipRule.tree_flatten,
        GlobalFlipRule.tree_unflatten
    )
except ValueError:
    pass

# B. Enregistrement Flax (Standard NetKet)
def _serialize_rule_flax(rule):
    return {"prob_global": rule.prob_global}

def _deserialize_rule_flax(rule, state_dict):
    return GlobalFlipRule(prob_global=state_dict["prob_global"])

try:
    serialization.register_serialization_state(
        GlobalFlipRule,
        _serialize_rule_flax,
        _deserialize_rule_flax
    )
except Exception:
    pass

# C. Enregistrement NQXPACK (Indispensable pour VOTRE environnement Pro)
# C'est ce bloc qui corrige l'erreur SerializationError que vous venez d'avoir.
try:
    from nqxpack._src.lib_v1 import register_serialization
    
    def _serialize_rule_nqx(rule):
        # On renvoie un dictionnaire simple
        return {"prob_global": rule.prob_global}

    # On enregistre (on ignore l'erreur si c'est déjà fait)
    try:
        register_serialization(GlobalFlipRule, _serialize_rule_nqx)
    except Exception:
        pass
        
except ImportError:
    # Si nqxpack n'est pas installé (ex: environnement standard), on ignore.
    pass