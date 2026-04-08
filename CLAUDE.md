# NeuralNetworkQuantumStates

## Acces distant
- Serveur: morbihan.polytechnique.fr
- User: max.anglade
- Chemin projet: /users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates
- Connexion: `ssh max.anglade@morbihan.polytechnique.fr` (authentification par cle)
- Toutes les commandes projet doivent etre executees via SSH sur ce serveur

## Description du projet
Projet de recherche sur les Neural Network Quantum States (NQS) appliques aux systemes de spins quantiques (modele d'Ising transverse).
Le but est d'utiliser des reseaux de neurones comme ansatz variationnels pour approximer les etats fondamentaux de hamiltoniens quantiques, en utilisant la methode VMC (Variational Monte Carlo).

Le projet comporte deux axes principaux :
1. **src/nqs_psc/** : Implementations classiques (RBM, CNN, Jastrow, Mean-Field, Boltzmann Machine) avec optimisation SGD et gradient naturel (NGD)
2. **Foundational/** : Approche plus avancee utilisant des Vision Transformers (ViTFNQS) avec le framework `netket_foundational`, permettant des modeles parametrises (Foundational NQS) qui generalisent sur un espace de parametres du Hamiltonien

## Stack technique
- Python 3.13
- NetKet 3.21.0 (framework VMC pour etats quantiques)
- JAX 0.7.2 + Flax 0.12.0 (differentiable programming)
- Optax 0.2.6 (optimiseurs)
- netket_pro 0.1 (installe en mode editable depuis ./netket_pro/)
- netket_foundational (sous-package de netket_pro, dans netket_pro/experiments/netket_foundational/)
- advanced_drivers (sous-package de netket_pro, callbacks pour drivers VMC)
- Matplotlib, Pandas, NumPy, SciPy, tqdm

## Structure du projet

### src/nqs_psc/ — Package principal
- `ansatz/ansatz.py` : Definitions des ansatz (MF, Jastrow, BM, CNN, CNN_exp) avec Flax/JAX
- `optimizer/optimizers.py` : SGD, NGD (gradient naturel), calcul de la matrice S (quantum Fisher), energies locales
- `utils/logging_utils.py` : Sauvegarde des runs (logs + meta.json)
- `train_base.py` : Script d'entrainement de base (Ising 2D, Hypercube L=4)
- `train_rbm*.py`, `train_cnn*.py` : Scripts d'entrainement specifiques par modele
- `plot/` : Scripts de visualisation (energie, Mz, Vscore, temps d'execution, etc.)
- `Fundationnal_models/` : Versions initiales des modeles foundational (1D_avec_desordre, Attention, Transformer)
- `animation/` : Scripts d'animation des configurations de spins

### Foundational/ — Experiences avancees (Foundational NQS)
Scripts d'entrainement et d'analyse utilisant le ViTFNQS (Vision Transformer Foundational NQS).
- `1D_sans_desordre.py` : Ising 1D, chaine propre, sweep en h (champ transverse)
- `1D_avec_desordre_mono_h0_et_exact.py` : Ising 1D avec desordre mono-site, comparaison exacte
- `1D_avec_desordre_pluri_h0_et_exact.py` : Ising 1D avec desordre pluri-sites, comparaison exacte
- `1D_avec_desordre_pluri_h0.py` : Ising 1D avec desordre pluri-sites
- `1D_avecc_desordre_pluri_h0_et_exact_chrono.py` : Version chronometree
- `ansatz_dual_embed.py` : Variante ViTFNQS avec double embedding (spins + couplages separes)
- `Ansatz Transformer.py` : Architecture Transformer complete (Embed, Encoder, OutputHead, FMHA)
- `compare_embeddings*.py` : Comparaison de strategies d'embedding
- `compare_epoch_training*.py` (v1 a v5b) : Comparaison du nombre d'epochs d'entrainement
- `test_post_training.py` : Tests post-entrainement
- `Plot/` : Scripts de visualisation specifiques (Mz distribution, vscore cmap, etc.)
- `logs/` : Runs sauvegardes (meta.json + state_*.nk checkpoints)
- `logs/Rami_FNQS/` : Runs specifiques au desordre (Foundational NQS)

### netket_pro/ — Dependance locale (installe en editable)
Fork/extension de NetKet Pro (Ecole Polytechnique / NeuralQXLab).
- `packages/` : Sous-packages (advanced_drivers, bridge, nqs_nets, ptvmc, spin_vmc, nqxpack, netket_checkpoint)
- `experiments/netket_foundational/` : Le framework foundational utilise dans Foundational/
  - `_src/model/vit.py` : Architecture ViTFNQS
  - `_src/model/attentions.py` : Mecanismes d'attention (FMHA)
  - `_src/vqs/state.py` : FoundationalQuantumState
  - `_src/hilbert/parameter_space.py` : ParameterSpace
  - `_src/operator/parametrized.py` : ParametrizedOperator
  - `_src/operator/pauli_strings/` : Operateurs Pauli (sigmax, sigmay, sigmaz)
  - `_src/operator/embed/` : Embedding d'operateurs discrets
  - `_src/operator/fermion2nd/` : Fermions 2nde quantification
  - `_src/driver/ngd/` : Driver VMC avec gradient naturel (SR/SRT)
  - `operator.py` : Exports publics (sigmax, sigmay, sigmaz, ParametrizedOperator)

### Racine
- `compare_dataset.py`, `fit_Mz_1D.py` : Scripts d'analyse de donnees
- `graphs/`, `logs/` : Resultats et logs des runs
- `Default Dataset.csv` : Dataset de reference

## Concepts physiques cles
- **Ising transverse** : H = -J sum(sigma_z_i * sigma_z_j) - h sum(sigma_x_i), transition de phase a h/J critique
- **VMC** : Methode variationnelle Monte Carlo — on echantillonne des configurations de spins via MCMC (MetropolisLocal), on estime l'energie et son gradient
- **NQS** : Le reseau de neurones encode log(psi(sigma)), la fonction d'onde log-amplitude
- **Gradient naturel (NGD/SR)** : Utilise la matrice de Fisher quantique S pour preconditionner le gradient (methode Stochastic Reconfiguration)
- **Foundational NQS (FNQS)** : Un seul modele (ViT) est entraine simultanement sur un espace de parametres du Hamiltonien (ParameterSpace), au lieu d'un modele par jeu de parametres. Utilise des replicas avec differentes valeurs de parametres.
- **Vscore** : Metrique de qualite de l'etat variationnel (variance normalisee)
- **Desordre** : Champs aleatoires locaux h_i appliques sur chaque site (mono = un seul site, pluri = plusieurs sites)
- **ViTFNQS** : Vision Transformer adapte aux FNQS — decoupe les spins en patches, embedding + attention multi-tete, output head

## Commandes utiles
```bash
# Se connecter au serveur
ssh max.anglade@morbihan.polytechnique.fr

# Aller au projet
cd /users/eleves-a/2024/max.anglade/Documents/NeuralNetworkQuantumStates

# Lancer un script Foundational (avec sharding JAX)
NETKET_EXPERIMENTAL_SHARDING=1 python Foundational/1D_sans_desordre.py

# Lancer un entrainement classique
python src/nqs_psc/train_rbm.py

# Voir les logs d'un run
cat Foundational/logs/run_*/meta.json

# Lire un fichier distant depuis Claude Code local
ssh max.anglade@morbihan.polytechnique.fr "cat <chemin>"

# Lister les fichiers distants
ssh max.anglade@morbihan.polytechnique.fr "ls -la <chemin>"
```

## Patterns importants

### Charger un FoundationalQuantumState depuis un checkpoint
```python
# 1. Charger le vstate (classmethod, reconstruit tout depuis le .nk)
vs = nkf.FoundationalQuantumState.load("logs/run_XXX/state_400.nk")
# Note: le module flip_rules doit etre importable pour la deserialisation du sampler
from flip_rules import GlobalFlipRule

# 2. L'hilbert serialise est un TensorGenericHilbert (spins+couplages),
#    il faut reconstruire le Spin hilbert pour les operateurs Pauli
L = meta["L"]  # depuis meta.json
hi = nk.hilbert.Spin(0.5, L)

# 3. Evaluer pour un jeu de parametres donne
_vs = vs.get_state(pars)  # pars = np.full(L, h0) ou vecteur de desordre
_vs.sample()
e = _vs.expect(hamiltonian)
```

### V-score
Utiliser la fonction deja implementee dans `spin_vmc` :
```python
from spin_vmc.data.funcs import vscore
v = vscore(E, VarE, N)  # V = N * VarE / E^2
```

### SR vs SRT (use_ntk)
- **SR** (`use_ntk=False`) : construit la matrice S = O^T O de taille `n_params x n_params` → explose en memoire pour grands modeles
- **SRT** (`use_ntk=True`) : construit K = O O^T de taille `n_samples x n_samples` → beaucoup plus petit quand n_params >> n_samples
- Pour L >= 36, toujours utiliser `use_ntk=True` dans `nkf.VMC_NG(...)`

## Conventions du projet
- Les fichiers .nk sont des checkpoints NetKet (etats sauvegardes)
- Les logs de run sont dans des dossiers horodates `run_YYYY-MM-DD_HH-MM-SS/`
- `meta.json` contient les hyperparametres et metadata de chaque run
- `test_results.csv` contient les resultats de test d'un run
- Le code est en Python, les commentaires souvent en francais
- Les scripts Foundational utilisent `netket_foundational` (import netket_foundational as nkf)
- Variable d'environnement `NETKET_EXPERIMENTAL_SHARDING=1` requise pour les scripts foundational
- Le repo git est sur GitHub: https://github.com/MaxAngl/NeuralNetworkQuantumStates
