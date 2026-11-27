"""Démos de l'expert planificateur dans le jumeau avec planche, EN PARALLÈLE (un processus par cœur).

    python avec_obstacle/il/record_demos_planche.py                 -> 1000 épisodes, tous les cœurs
    python avec_obstacle/il/record_demos_planche.py 300 --proc 4    -> 300 épisodes, 4 processus
    python avec_obstacle/il/record_demos_planche.py 1000 --dart 0   -> sans bruit
    python avec_obstacle/il/record_demos_planche.py 2000 --depart 3000   -> 2000 AUTRES scènes (seeds 3000..4999),
                                                         écrites dans demos_planche_3000.npz : le BC
                                                         prend tous les fichiers demos_planche*.npz

Seeds à partir de 1000 (l'évaluation utilise 0..N, jamais vues ici). Seuls les épisodes réussis sont
gardés. DART : sur la moitié des épisodes, on ajoute du bruit aux actions EXÉCUTÉES (segments libres
seulement : approche, transport ; pas près du cube ni de la table) mais on enregistre l'action propre
de l'expert -> les démos montrent aussi comment revenir sur le plan après un écart (c'est ce qui
manquait au BC pur, cf. DAgger dans l'ancien pipeline).

Sorties (dans il/) :
    demos_planche.npz          obs, act, rew, next_obs, done, episode  (rew/next_obs/done : pour le SAC)
    demos_planche_stats.json   par épisode : scène, temps de planification, durée, chemin, gardé ou non
"""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import numpy as np  # noqa: E402

SEED_OFFSET = int(sys.argv[sys.argv.index("--depart") + 1]) if "--depart" in sys.argv else 1000
SUFFIXE = "" if SEED_OFFSET == 1000 else f"_{SEED_OFFSET}"
DART_SIGMA = float(sys.argv[sys.argv.index("--dart") + 1]) if "--dart" in sys.argv else 0.15
PAS_APRES = 15            # pas enregistrés après la fin du plan (le robot reste immobile, cube posé)

_env = _expert = None


def _init():
    global _env, _expert
    from expert_planche import ExpertPlanche
    from twin_env import TwinPlancheEnv
    _env, _expert = TwinPlancheEnv(), ExpertPlanche()


def episode(seed):
    from twin_env import MAX_STEPS
    env, ex = _env, _expert
    rng = np.random.default_rng(seed + 12345)
    obs, _ = env.reset(seed=seed)
    t0 = time.time()
    ok_plan = ex.reset(env)
    stats = {"seed": seed, "posture": env.scene["posture"], "sur_chemin": env.scene["sur_chemin"],
             "plan": ok_plan, "message": ex.message, "t_plan": round(time.time() - t0, 2)}
    if not ok_plan:
        return stats, None
    sigma = DART_SIGMA if rng.random() < 0.5 else 0.0
    O, A, R, O2, D = [], [], [], [], []
    apres, info = 0, {}
    for _ in range(MAX_STEPS):
        a = ex.action(env)
        a_exec = a.copy()
        if sigma > 0 and not ex.fini() and not ex.precis:      # bruit seulement sur les segments libres
            a_exec[:7] = np.clip(a_exec[:7] + rng.normal(0, sigma, 7), -1, 1)
        obs2, r, term, trunc, info = env.step(a_exec)
        O.append(obs)
        A.append(a)
        R.append(r)
        O2.append(obs2)
        D.append(term)
        obs = obs2
        if term or trunc:
            break
        if ex.fini():
            apres += 1
            if apres >= PAS_APRES:
                break
    stats.update(succes=bool(info.get("is_success")), collision=info.get("collision", ""),
                 pas=info.get("pas"), longueur_q=info.get("longueur_q"), dart=sigma)
    if not stats["succes"]:
        return stats, None
    return stats, (np.array(O, np.float32), np.array(A, np.float32), np.array(R, np.float32),
                   np.array(O2, np.float32), np.array(D, bool))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1000
    proc = int(sys.argv[sys.argv.index("--proc") + 1]) if "--proc" in sys.argv else (os.cpu_count() or 2)
    print(f"{n} épisodes, {proc} processus, bruit DART {DART_SIGMA} sur la moitié des épisodes")
    t0 = time.time()
    stats, data = [], []
    with Pool(proc, initializer=_init) as pool:
        for k, (st, d) in enumerate(pool.imap_unordered(episode, range(SEED_OFFSET, SEED_OFFSET + n))):
            stats.append(st)
            if d is not None:
                data.append((st["seed"], d))
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{n} joués, {len(data)} gardés  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    data.sort(key=lambda x: x[0])
    O = np.concatenate([d[0] for _, d in data])
    A = np.concatenate([d[1] for _, d in data])
    R = np.concatenate([d[2] for _, d in data])
    O2 = np.concatenate([d[3] for _, d in data])
    D = np.concatenate([d[4] for _, d in data])
    E = np.concatenate([np.full(len(d[0]), i) for i, (_, d) in enumerate(data)])
    np.savez_compressed(f"demos_planche{SUFFIXE}.npz", obs=O, act=A, rew=R, next_obs=O2, done=D, episode=E)
    with open(f"demos_planche{SUFFIXE}_stats.json", "w") as f:
        json.dump(sorted(stats, key=lambda s: s["seed"]), f, indent=1)
    sans_plan = sum(not s["plan"] for s in stats)
    rates = sum(s["plan"] and not s.get("succes") for s in stats)
    print(f"\ndemos_planche{SUFFIXE}.npz : {len(data)} épisodes gardés, {len(O)} pas "
          f"({sans_plan} scènes sans plan possible, {rates} plans ratés à l'exécution)")
    print(f"Temps total : {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
