from pathlib import Path
import json
import time
import numpy as np
import re
import os
from datetime import datetime

try:
    import jax.numpy as jnp
except Exception:
    jnp = None


def _to_jsonable(x):
    """
    Convertit récursivement n'importe quel objet en structure JSON-safe
    (dict / list / str / int / float / bool / None).

    Hypothèse physique : les quantités vraiment pertinentes (énergie, etc.)
    sont réelles. Si un complexe apparaît, on prend simplement la partie réelle.
    """

    # --- JAX DeviceArray : on repasse par NumPy ---
    if jnp is not None and isinstance(x, jnp.ndarray):
        x = np.asarray(x)

    # --- scalaires Python complexes ---
    if isinstance(x, complex):
        return float(np.real(x))

    # --- scalaires NumPy (float64, int64, complex128, ...) ---
    if isinstance(x, np.generic):
        val = x.item()
        if isinstance(val, complex):
            return float(np.real(val))
        return val

    # --- tableaux NumPy ---
    if isinstance(x, np.ndarray):
        # si tableau complexe → on garde la partie réelle
        if np.iscomplexobj(x):
            return np.real(x).tolist()
        return x.tolist()

    # --- dict récursif ---
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    # --- listes / tuples récursifs ---
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    # --- objets avec .to_dict() ---
    if hasattr(x, "to_dict") and callable(x.to_dict):
        return _to_jsonable(x.to_dict())

    # --- types JSON natifs ---
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    # --- fallback : représentation texte (pour objets "exotiques") ---
    return str(x)


def sanitize(text: str) -> str:
    """Nettoie les chaînes pour les noms de dossiers."""
    text = str(text)
    text = re.sub(r"[^\w\-=.]+", "_", text)
    return text.strip("_")


def save_runtime_log(logger, run_root="runs", meta=None):
    """
    Sauvegarde :
      - metrics.runtime.json : contenu de logger.data (ou logger si pas d'attribut .data)
      - meta.json : dictionnaire meta

    Compatible avec l'ancienne API :
      - logger est un RuntimeLog avec attribut .data
      - plot.py peut faire List_logger[i]["Energy"]["iters"] sans rien changer
    """
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    meta = meta or {}
    meta_jsonable = _to_jsonable(meta)

    # Construction du nom du dossier à partir de meta
    name_parts = []
    for key in ["model", "hamiltonien", "L", "n", "lr", "diag_shift", "N_e"]:
        if key in meta_jsonable:
            val = meta_jsonable[key]
            name_parts.append(f"{key}={sanitize(val)}")

    run_name = "_".join(name_parts) if name_parts else "run"
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Récupération des données du logger (RuntimeLog.data ou logger brut)
    raw_data = getattr(logger, "data", logger)
    data_jsonable = _to_jsonable(raw_data)

    # Sauvegarde des JSON (types déjà JSON-safe → pas besoin de default=)
    (run_dir / "metrics.runtime.json").write_text(
        json.dumps(data_jsonable, indent=2),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(meta_jsonable, indent=2),
        encoding="utf-8",
    )

    return run_dir


import numpy as np


import numpy as np
from netket.utils.history import History


import numpy as np
from netket.utils.history import History


def to_jsonable(obj):
    """
    Convertit toutes les structures possibles de RuntimeLog NetKet en JSON.
    Complexes -> partie réelle.
    """

    # 1) ----- NetKet History -----
    if isinstance(obj, History):
        # Cas 1 : History possède .data
        if hasattr(obj, "data") and isinstance(obj.data, dict):
            return {k: to_jsonable(v) for k, v in obj.data.items()}

        # Cas 2 : History possède .values() comme dict-like
        try:
            values = obj.values()  # fonctionne pour History(keys=['value'])
            if hasattr(values, "items"):
                return {k: to_jsonable(v) for k, v in values.items()}
        except Exception:
            pass

        # Cas 3 : History est juste une séquence
        try:
            return to_jsonable(list(obj))
        except Exception:
            pass

        # Cas 4 : fallback extrême : conversion string (ne crash jamais)
        return str(obj)

    # 2) ----- numpy scalaires -----
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)

    # 3) ----- complexes -----
    if isinstance(obj, (np.complexfloating, complex)):
        return float(obj.real)

    # 4) ----- scalaires Python -----
    if isinstance(obj, (int, float, bool, str, type(None))):
        return obj

    # 5) ----- numpy array -----
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]

    # 6) ----- JAX DeviceArray -----
    try:
        import jax.numpy as jnp

        if isinstance(obj, jnp.ndarray):
            return [to_jsonable(x) for x in np.array(obj)]
    except Exception:
        pass

    # 7) ----- list / tuple -----
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # 8) ----- dict -----
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # 9) ----- dernier fallback : conversion string -----
    return str(obj)


def runtime_log_to_jsonable(log_obj):
    return {key: to_jsonable(series) for key, series in log_obj.data.items()}


def save_run(log, meta: dict, base_dir="logs"):
    """
    Sauvegarde un run NetKet :
      - crée un dossier horodaté dans base_dir
      - écrit meta.json et log.json
    `meta` doit être passé depuis l'extérieur (liberté totale pour le construire).
    """
    # Dossier horodaté
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Sauvegarde META
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Sauvegarde LOG
    json_log = runtime_log_to_jsonable(log)
    with open(os.path.join(run_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=2)

    return run_dir
