"""Évaluation commune (expert, BC, SAC) sur les MÊMES scènes avec planche.

Les scènes d'évaluation sont les seeds 0..N-1 (les démos utilisent les seeds >= 1000, jamais vues).
Une scène où l'expert ne trouve aucun plan est infaisable avec une prise par le dessus (ex. poteau
entre la base et le cube) : elle est écartée pour tout le monde. La liste des scènes faisables est
gardée dans results/scenes_eval_planche.json (calculée une fois).

    from eval_planche import evaluer, scenes_eval
    res = evaluer(lambda env, obs: action, n=100)
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin_env import MAX_STEPS, TwinPlancheEnv  # noqa: E402

ROOT = os.path.join(HERE, "..")
FICHIER = os.path.join(ROOT, "results", "scenes_eval_planche.json")
PAS_STABLE = 25           # réussite tenue 1 s -> épisode terminé (réussi)


def scenes_eval(n):
    """Seeds 0..n-1 faisables pour l'expert (calcul une fois, puis relu)."""
    deja = {}
    if os.path.exists(FICHIER):
        with open(FICHIER) as f:
            deja = {int(k): v for k, v in json.load(f).items()}
    manquants = [s for s in range(n) if s not in deja]
    if manquants:
        from expert_planche import ExpertPlanche
        env, ex = TwinPlancheEnv(), ExpertPlanche()
        print(f"Préparation des scènes d'évaluation ({len(manquants)} à tester avec l'expert)...")
        for s in manquants:
            env.reset(seed=s)
            deja[s] = {"faisable": bool(ex.reset(env)), "scene": env.scene, "q0": env.q0.tolist()}
        os.makedirs(os.path.dirname(FICHIER), exist_ok=True)
        with open(FICHIER, "w") as f:
            json.dump({str(k): v for k, v in sorted(deja.items())}, f)
    return [(s, deja[s]) for s in range(n) if deja[s]["faisable"]]


def jouer(env, politique, seed, sc, reset_cb=None, view=False):
    """Un épisode sur la scène sc. politique(env, obs) -> action. Renvoie les mesures."""
    obs, _ = env.reset(seed=seed, options={"scene": sc["scene"], "q0": sc["q0"]})
    if reset_cb is not None:
        reset_cb(env)
    stable, t_succes, info = 0, None, {}
    for _ in range(MAX_STEPS):
        obs, _, term, trunc, info = env.step(politique(env, obs))
        if view:
            time.sleep(0.02)
        if info["is_success"]:
            stable += 1
            t_succes = t_succes or info["pas"]
        else:
            stable, t_succes = 0, None
        if term or trunc or stable >= PAS_STABLE:
            break
    s = env.scene
    return {"seed": seed, "posture": s["posture"], "sur_chemin": s["sur_chemin"],
            "succes": bool(info.get("is_success", False)), "collision": info.get("collision", ""),
            "pas": t_succes, "longueur_q": info.get("longueur_q")}


def evaluer(politique, n=100, obs_planche=True, reset_cb=None, view=False, nom=""):
    env = TwinPlancheEnv(render_mode="human" if view else None, obs_planche=obs_planche)
    res = []
    scenes = scenes_eval(n)
    t0 = time.time()
    for k, (seed, sc) in enumerate(scenes[:5] if view else scenes):
        r = jouer(env, politique, seed, sc, reset_cb, view)
        res.append(r)
        if view or (k + 1) % 20 == 0:
            ok = sum(x["succes"] for x in res)
            print(f"  {nom} {k + 1}/{len(scenes)} scènes : {ok} réussies ({time.time() - t0:.0f} s)")
    env.close()
    return res


def resume(res, nom=""):
    """Taux de réussite, collisions, longueur et durée des réussites ; détail par posture."""
    def ligne(rs):
        n = len(rs)
        if n == 0:
            return "-"
        ok = [r for r in rs if r["succes"]]
        col = sum(bool(r["collision"]) for r in rs)
        lon = np.mean([r["longueur_q"] for r in ok]) if ok else float("nan")
        dur = np.mean([r["pas"] for r in ok]) * 0.04 if ok else float("nan")
        return f"{len(ok):3d}/{n:<3d} ({100 * len(ok) / n:3.0f} %)  collisions {col:3d}  chemin {lon:5.2f} rad  durée {dur:4.1f} s"
    print(f"\n=== {nom} ===")
    print(f"  {'toutes':22s} {ligne(res)}")
    for po in ("plat", "tranche", "debout"):
        for ch in (True, False):
            rs = [r for r in res if r["posture"] == po and r["sur_chemin"] == ch]
            print(f"  {po:8s} {'sur le chemin' if ch else 'ailleurs':13s} {ligne(rs)}")
