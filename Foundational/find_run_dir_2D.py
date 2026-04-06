"""
Trouve le run_dir le plus récent correspondant à L et n_dim=2 dans logs/.
Usage: python find_run_dir_2D.py <L>
Affiche le chemin absolu du run_dir, ou rien (exit 1) si non trouvé.
"""
import json
import glob
import os
import sys

L = int(sys.argv[1])
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
candidates = sorted(
    glob.glob(os.path.join(logs_dir, "run_*/meta.json")) +
    glob.glob(os.path.join(logs_dir, "*/run_*/meta.json")) +
    glob.glob(os.path.join(logs_dir, "*/L=*/meta.json")),
    reverse=True
)

for meta_path in candidates:
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("L") == L and meta.get("n_dim") == 2:
            print(os.path.dirname(meta_path))
            sys.exit(0)
    except Exception:
        pass

print(f"[find_run_dir_2D] Aucun run 2D avec L={L} trouvé dans {logs_dir}", file=sys.stderr)
sys.exit(1)
