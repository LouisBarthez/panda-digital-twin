"""Données pour apprendre les DÉCISIONS (orientation de pince, route d'approche, route de transport).

Pour chaque scène tirée au hasard, toutes les options sont évaluées dans le jumeau (faisable sans
collision ? coût en radians ?). L'« expert discret » choisit la moins chère : c'est l'étiquette de
l'IL. Les tableaux complets servent à l'analyse (le choix du réseau était-il faisable ? optimal ?).

    python avec_obstacle/il/record_decisions_planche.py              -> 3000 scènes (seeds 1000..3999), tous les cœurs
    python avec_obstacle/il/record_decisions_planche.py 5000 --proc 8

Sortie : il/decisions_planche.npz
    feat (N, 23)  entrées du réseau          label (N, 3) k, route approche, route transport (-1 : aucune option)
    ok_k (N, 5), ok_app / ok_tr (N, 5, 6), c_app / c_tr (N, 5, 6), c_fixe (N, 5), posture, sur_chemin, seed
"""
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import numpy as np  # noqa: E402

DEPART = int(sys.argv[sys.argv.index("--depart") + 1]) if "--depart" in sys.argv else 1000
_env = _gen = None


def _init():
    global _env, _gen
    from decisions_planche import Generateur
    from twin_env import TwinPlancheEnv
    _env, _gen = TwinPlancheEnv(), Generateur()


def scene(seed):
    from decisions_planche import caracteristiques
    from planche import POSTURES
    _env.reset(seed=seed)
    det, q0 = _env.detection(), _env.q_cmd.copy()
    o = _gen.evaluer_options(det, q0)
    lab = np.array(o["meilleur"] if o["meilleur"] is not None else (-1, -1, -1))
    return {"feat": caracteristiques(det, q0), "label": lab, "ok_k": o["ok_k"], "ok_app": o["ok_app"],
            "ok_tr": o["ok_tr"], "c_app": o["c_app"], "c_tr": o["c_tr"], "c_fixe": o["c_fixe"],
            "posture": POSTURES.index(_env.scene["posture"]), "sur_chemin": _env.scene["sur_chemin"],
            "seed": seed}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3000
    proc = int(sys.argv[sys.argv.index("--proc") + 1]) if "--proc" in sys.argv else (os.cpu_count() or 2)
    print(f"{n} scènes, {proc} processus")
    t0 = time.time()
    res = []
    with Pool(proc, initializer=_init) as pool:
        for i, r in enumerate(pool.imap_unordered(scene, range(DEPART, DEPART + n))):
            res.append(r)
            if (i + 1) % 100 == 0:
                ok = sum(x["label"][0] >= 0 for x in res)
                print(f"  {i + 1}/{n} scènes, {ok} avec au moins une option ({(time.time() - t0) / 60:.1f} min)",
                      flush=True)
    res.sort(key=lambda r: r["seed"])
    out = {k: np.array([r[k] for r in res]) for k in res[0]}
    suffixe = "" if DEPART == 1000 else f"_{DEPART}"
    np.savez_compressed(f"decisions_planche{suffixe}.npz", **out)
    ok = out["label"][:, 0] >= 0
    print(f"\ndecisions_planche{suffixe}.npz : {len(res)} scènes, {ok.sum()} avec une option faisable "
          f"({(time.time() - t0) / 60:.1f} min)")
    from decisions_planche import ROUTES
    for nom, col in (("approche", 1), ("transport", 2)):
        cpt = np.bincount(out["label"][ok, col], minlength=len(ROUTES))
        print(f"  routes choisies par l'expert ({nom}) : " + ", ".join(f"{r} {c}" for r, c in zip(ROUTES, cpt)))


if __name__ == "__main__":
    main()
