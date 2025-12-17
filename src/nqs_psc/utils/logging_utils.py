from pathlib import Path
import json
import time
import numpy as np
import re
import os
from datetime import datetime


def save_run(log, meta: dict, base_dir="logs", create_only=False, run_dir=None):
    """
    Sauvegarde les logs et metadata d'un run.
    
    Args:
        log: Logger NetKet
        meta: Dictionnaire de métadonnées
        base_dir: Dossier de base pour les runs
        create_only: Si True, crée seulement le dossier et sauvegarde les meta, pas les logs
        run_dir: Si fourni, utilise ce dossier au lieu d'en créer un nouveau
    
    Returns:
        Chemin du dossier de run
    """
    
    # Créer ou utiliser le dossier
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)

    # Sauvegarde META
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Sauvegarde LOG (seulement si pas en mode create_only)
    if not create_only:
       log.serialize(os.path.join(run_dir, "log.json"))

    return run_dir
