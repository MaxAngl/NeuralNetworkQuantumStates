import os
os.environ["NETKET_EXPERIMENTAL_SHARDING"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk
import netket_foundational as nkf
import numpy as np
import json
from spin_vmc.data.funcs import vscore
from flip_rules import GlobalFlipRule

run_dir = "../logs/run_2026-03-05_11-31-03"
vs = nkf.FoundationalQuantumState.load(f"{run_dir}/state_400.nk")

with open(f"{run_dir}/meta.json") as f:
    meta = json.load(f)

L = meta["L"]
hi = nk.hilbert.Spin(0.5, L)
J_val = 1.0

def create_operator(params):
    ha_X = sum(params[i] * nkf.operator.sigmax(hi, i) for i in range(hi.size))
    ha_ZZ = sum(nkf.operator.sigmaz(hi, i) @ nkf.operator.sigmaz(hi, (i+1) % hi.size) for i in range(hi.size))
    return -ha_X - J_val * ha_ZZ

test_h0 = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0]
print(f"L = {L}")
print(f"{'h0':>6} | {'Energy':>12} | {'Variance':>12} | {'V_score':>12}")
print("-" * 55)
for h0 in test_h0:
    pars = np.full(L, h0)
    _ha = create_operator(pars)
    _vs = vs.get_state(pars)
    _vs.sample()
    e = _vs.expect(_ha)
    vscore_val = vscore(e.mean.real, e.variance.real, L)
    print(f"{h0:6.2f} | {e.mean.real:12.4f} | {e.variance.real:12.4f} | {vscore_val:12.6e}")
