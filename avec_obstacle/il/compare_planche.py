"""Comparaison expert / BC / BC sans planche / SAC / SAC + démos sur les MÊMES scènes avec planche.

    python avec_obstacle/il/compare_planche.py            -> 100 scènes (seeds 0..99 faisables), tout ce qui est entraîné
    python avec_obstacle/il/compare_planche.py --n 200
    python avec_obstacle/il/compare_planche.py --figure   -> refait seulement la figure à partir du JSON

Méthodes prises si leur fichier existe : il/bc_planche.pt, il/bc_dagger_planche.pt, il/bc_sans_planche.pt,
rl/sac_planche_best.zip, rl/sac_planche_demos_best.zip. L'expert est toujours évalué (référence).
Sorties : results/compare_planche.json et results/compare_planche.png (pour le rapport).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from eval_planche import evaluer, resume  # noqa: E402

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
CATS = [(po, ch) for po in ("plat", "tranche", "debout") for ch in (True, False)]


def methodes():
    from expert_planche import ExpertPlanche
    ex = ExpertPlanche()
    yield "expert (planificateur)", (lambda env, obs: ex.action(env)), True, (lambda env: ex.reset(env))
    import train_bc_planche as bc
    for nom, lib, obs_pl in (("bc_planche", "BC", True), ("bc_dagger_planche", "BC + DAgger", True),
                             ("bc_sans_planche", "BC sans planche (ablation)", False)):
        if os.path.exists(os.path.join(HERE, f"{nom}.pt")):
            yield lib, bc.fonction(bc.charger(nom)), obs_pl, None
    for nom, lib in (("sac_planche", "SAC"), ("sac_planche_demos", "SAC + démos")):
        z = os.path.join(ROOT, "rl", f"{nom}_best.zip")
        if os.path.exists(z):
            from stable_baselines3 import SAC
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from twin_env import TwinPlancheEnv
            venv = VecNormalize.load(os.path.join(ROOT, "rl", f"vecnorm_{nom}_best.pkl"),
                                     DummyVecEnv([TwinPlancheEnv]))
            venv.training = False
            model = SAC.load(z)

            def f(env, obs, model=model, venv=venv):
                a, _ = model.predict(venv.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
                return a[0]
            yield lib, f, True, None


def figure(tous, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noms = list(tous)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(CATS))
    w = 0.8 / len(noms)
    for i, nom in enumerate(noms):
        taux, lon = [], []
        for po, ch in CATS:
            rs = [r for r in tous[nom] if r["posture"] == po and r["sur_chemin"] == ch]
            ok = [r for r in rs if r["succes"]]
            taux.append(100 * len(ok) / len(rs) if rs else 0)
            lon.append(np.mean([r["longueur_q"] for r in ok]) if ok else 0)
        axes[0].bar(x + i * w, taux, w, label=nom)
        axes[1].bar(x + i * w, lon, w, label=nom)
    etiq = [f"{po}\n{'sur le chemin' if ch else 'ailleurs'}" for po, ch in CATS]
    for ax, t in zip(axes, ("Réussite (%)", "Chemin articulaire des réussites (rad)")):
        ax.set_xticks(x + w * (len(noms) - 1) / 2)
        ax.set_xticklabels(etiq, fontsize=8)
        ax.set_title(t)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(0, 105)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(noms), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(chemin, dpi=130)
    print(f"Figure : {chemin}")


def main():
    if "--figure" in sys.argv:
        with open(os.path.join(ROOT, "results", "compare_planche.json")) as fj:
            figure(json.load(fj), os.path.join(ROOT, "results", "compare_planche.png"))
        return
    tous = {}
    for nom, f, obs_pl, cb in methodes():
        print(f"\nÉvaluation : {nom}")
        tous[nom] = evaluer(f, n=N, obs_planche=obs_pl, reset_cb=cb, nom=nom)
        resume(tous[nom], nom)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "compare_planche.json"), "w") as fj:
        json.dump(tous, fj, indent=1)
    figure(tous, os.path.join(ROOT, "results", "compare_planche.png"))


if __name__ == "__main__":
    main()
