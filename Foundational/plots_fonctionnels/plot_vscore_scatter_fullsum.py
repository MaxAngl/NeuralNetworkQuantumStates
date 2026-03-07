import os
import sys
import glob
import json
import numpy as np
import jax
import netket as nk
import netket_foundational as nkf
from netket_foundational._src.model.vit import ViTFNQS
import flax
import msgpack
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import zipfile
import jax.numpy as jnp

# ==========================================
# 1. CONFIGURATION
# ==========================================
RUN_DIR = r"/users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates/logs/tests_echantillonnage_L=16/run_2026-03-02_20-12-16"

H0_TEST_LIST = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5, 2.5, 3.5, 4.5]
N_TEST_PER_H0 = 20
prob_global_flip = 0.01

# ==========================================
# 2. SETUP
# ==========================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
foundational_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, foundational_dir)
sys.path.insert(0, project_root)
from flip_rules import GlobalFlipRule

class SafeGlobalFlipRule(GlobalFlipRule):
    def transition(self, sampler, machine, parameters, state, key, sigma):
        sigma_new, log_prob = super().transition(sampler, machine, parameters, state, key, sigma)
        return jnp.asarray(sigma_new, dtype=sigma.dtype), log_prob

if not os.path.exists(RUN_DIR):
    print(f"Erreur : Le dossier {RUN_DIR} n existe pas.")
    sys.exit(1)

meta_path = os.path.join(RUN_DIR, "meta.json")
with open(meta_path, 'r') as f:
    meta = json.load(f)

L = meta["L"]
J_val = meta["hamiltonian"]["J"]
sigma_disorder = meta["hamiltonian"]["sigma"]
h0_train_list = meta["hamiltonian"]["h0_train_list"]
vit_params = meta["vit_config"]

hi = nk.hilbert.Spin(0.5, L)
ps = nkf.ParameterSpace(N=hi.size, min=0, max=10*max(H0_TEST_LIST + h0_train_list))

ma = ViTFNQS(
    num_layers=vit_params["num_layers"],
    d_model=vit_params["d_model"],
    heads=vit_params["heads"],
    b=vit_params["b"],
    L_eff=vit_params["L_eff"],
    n_coups=ps.size,
    complex=True,
    disorder=True,
    transl_invariant=False,
    two_dimensional=False,
)

sa = nk.sampler.MetropolisSampler(hi, rule=SafeGlobalFlipRule(prob_global_flip), n_chains=1)
vs = nkf.FoundationalQuantumState(sa, ma, ps, n_replicas=1, n_samples=1, seed=1)

checkpoints = glob.glob(os.path.join(RUN_DIR, "*.nk"))
if not checkpoints:
    print("Aucun checkpoint (.nk) trouve.")
    sys.exit(1)
last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]

# ==========================================
# 3. CHARGEMENT VIA ZIPFILE (IDENTIQUE)
# ==========================================
print(f"Chargement depuis archive : {os.path.basename(last_checkpoint)}")

state_dict = None

if not zipfile.is_zipfile(last_checkpoint):
    print("Ce n est pas un fichier ZIP. Tentative de chargement Raw MsgPack...")
    with open(last_checkpoint, 'rb') as f:
        state_dict = flax.serialization.msgpack_restore(f.read())
else:
    with zipfile.ZipFile(last_checkpoint, 'r') as zf:
        file_list = zf.namelist()
        candidates = [f for f in file_list if f.endswith('.msgpack')]
        if not candidates:
             print(f"Erreur : Aucun fichier .msgpack trouve dans archive ZIP. Contenu : {file_list}")
             sys.exit(1)
        target_file = sorted(candidates, key=len)[-1]
        print(f"   -> Extraction de : {target_file}")
        with zf.open(target_file) as f:
            state_dict = flax.serialization.msgpack_restore(f.read())

if state_dict is None:
    print("Echec du chargement des donnees.")
    sys.exit(1)

vars_dict = None
if 'variables' in state_dict:
    vars_dict = state_dict['variables']
elif 'model' in state_dict and 'variables' in state_dict['model']:
    vars_dict = state_dict['model']['variables']
elif 'params' in state_dict:
    vars_dict = state_dict
elif 'vqs' in state_dict and 'variables' in state_dict['vqs']:
    vars_dict = state_dict['vqs']['variables']

if vars_dict is None:
    vars_dict = state_dict

try:
    vs.variables = flax.serialization.from_state_dict(vs.variables, vars_dict)
    print("Poids injectes avec succes !")
except Exception as e:
    print(f"Erreur lors de injection Flax : {e}")
    print("Cles trouvees :", state_dict.keys() if isinstance(state_dict, dict) else "Not a dict")
    sys.exit(1)

# ==========================================
# 4. COMPUTE VSCORE - VERSION FULLSUM
# ==========================================
def create_operator(params):
    ha_X = sum(params[i] * nk.operator.spin.sigmax(hi, i) for i in range(L))
    ha_ZZ = sum(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, (i + 1) % L) for i in range(L))
    return -ha_X - J_val * ha_ZZ

def compute_vscore(params_batch):
    """Version FullSum : somme exacte sur tout espace de Hilbert."""
    v_scores = []

    for pars in params_batch:
        H = create_operator(pars)

        _vs = vs.get_state(pars)
        fs = nk.vqs.FullSumState(
            hilbert=hi,
            model=_vs.model,
            variables=_vs.variables,
        )

        res = fs.expect(H)
        mean_E = res.Mean.real
        var_E = res.variance
        val_score = var_E / (mean_E**2 + 1e-12)
        v_scores.append(val_score)

    return np.array(v_scores)

print("\nTraitement des donnees TRAIN...")
train_h0 = []
train_v_scores = []
disorder_file = os.path.join(RUN_DIR, "disorder.configs.npy")
if not os.path.exists(disorder_file):
    disorder_file = os.path.join(RUN_DIR, "disorder_configs.npy")

if os.path.exists(disorder_file):
    train_params = np.load(disorder_file)
    train_params_list = list(train_params)
    train_v_scores = compute_vscore(tqdm(train_params_list, desc="V-score Train"))
    n_h0_train = len(h0_train_list)
    replicas_per_h0_train = len(train_params) // n_h0_train
    for h0 in h0_train_list:
        train_h0.extend([h0] * replicas_per_h0_train)
else:
    print("Fichier disorder_configs.npy introuvable.")

print("\nTraitement des donnees TEST...")
test_h0 = []
test_v_scores = []
rng = np.random.default_rng(seed=42)
for h0 in tqdm(H0_TEST_LIST, desc="Generation Test"):
    test_configs = np.abs(rng.normal(loc=h0, scale=sigma_disorder, size=(N_TEST_PER_H0, L)))
    test_configs_list = list(test_configs)
    scores = compute_vscore(test_configs_list)
    test_h0.extend([h0] * N_TEST_PER_H0)
    test_v_scores.extend(scores)

# ==========================================
# 6. PLOTTING
# ==========================================
plt.figure(figsize=(10, 6))

plt.scatter(
    test_h0, test_v_scores,
    color='crimson', alpha=0.3, s=40,
    label='Test Replicas', edgecolor='none', marker='o', zorder=1
)

if len(train_h0) > 0:
    plt.scatter(
        train_h0, train_v_scores,
        color='royalblue', alpha=0.3, s=40,
        label='Train Replicas', edgecolor='none', marker='^', zorder=1
    )

df_test = pd.DataFrame({"h0": test_h0, "v": test_v_scores})
mean_test = df_test.groupby("h0")["v"].mean()

plt.plot(mean_test.index, mean_test.values, color='red', linestyle='--', linewidth=1.5, label='Test Mean', zorder=2)
plt.scatter(mean_test.index, mean_test.values, color='red', marker='s', s=60, edgecolor='black', zorder=3)

if len(train_h0) > 0:
    df_train = pd.DataFrame({"h0": train_h0, "v": train_v_scores})
    mean_train = df_train.groupby("h0")["v"].mean()
    plt.plot(mean_train.index, mean_train.values, color='blue', linestyle='--', linewidth=1.5, label='Train Mean', zorder=2)
    plt.scatter(mean_train.index, mean_train.values, color='blue', marker='s', s=60, edgecolor='black', zorder=3)

plt.yscale('log')
plt.xlabel(r"Transverse Field $h_0$", fontsize=12)
plt.ylabel(r"V-score ($\text{Var}(E) / E^2$)", fontsize=12)
plt.title(f"Accuracy Landscape FULLSUM: Train vs Test (L={L})", fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(loc='best', frameon=True, fontsize=10)

output_file = os.path.join(RUN_DIR, f"vscore_scatter_FULLSUM_L={L}.pdf")
plt.tight_layout()
plt.savefig(output_file)
plt.savefig(output_file.replace(".pdf", ".png"), dpi=150)
print(f"\nPlot sauvegarde : {output_file}")
print(f"Plot sauvegarde : {output_file.replace('.pdf', '.png')}")
