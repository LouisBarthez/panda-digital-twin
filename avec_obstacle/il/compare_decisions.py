"""Comparaison des politiques de DÉCISION sur les mêmes scènes que l'expert et le BC.

    python avec_obstacle/il/compare_decisions.py            -> 100 scènes (seeds 0..99 faisables)
    python avec_obstacle/il/compare_decisions.py --figure   -> refait seulement la figure
    python avec_obstacle/il/compare_decisions.py --seulement IL,RL   -> seulement les méthodes dont le nom contient IL ou RL

Pour IL et RL, 3 modes (voir MODES dans common/decisions_planche.py) : brut (la politique seule),
+ accessible (le générateur refuse les orientations de pince impossibles), + jumeau (les choix sont
testés dans le jumeau dans l'ordre de la politique, le 1er sûr est exécuté).

Méthodes (toutes exécutées par le même générateur, sauf la référence) :
    planificateur complet   la référence (RRT, cherche son chemin) — repris de results/compare_planche.json
    expert discret          le meilleur des 180 choix, calculé en testant tout (étiquette de l'IL)
    toujours tout droit     pince la moins tournée, approche et transport en ligne droite (ignore la planche)
    hasard                  un choix au hasard parmi les 180
    IL / IL sans planche    il/decision_il.pt, il/decision_il_sans_planche.pt
    RL                      rl/decision_rl.pt
Sorties : results/compare_decisions.json et .png
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))

import numpy as np  # noqa: E402

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
CATS = [(po, ch) for po in ("plat", "tranche", "debout") for ch in (True, False)]
SORTIE = os.path.join(ROOT, "results", "compare_decisions")


def methodes():
    from decisions_planche import N_K, N_R, Generateur
    gen = Generateur()

    def expert_discret(feat, det, q0, env):
        o = gen.evaluer_options(det, q0)
        return o["meilleur"] if o["meilleur"] is not None else (2, 0, 0)
    yield "expert discret", expert_discret, "brut"
    yield "toujours tout droit", (lambda feat, det, q0, env: (2, 0, 0)), "brut"
    rng = np.random.default_rng(0)
    yield "hasard", (lambda feat, det, q0, env: (int(rng.integers(N_K)), int(rng.integers(N_R)),
                                                 int(rng.integers(N_R)))), "brut"
    import train_decision_il as il
    for nom, lib, sans in (("decision_il", "IL", False), ("decision_il_sans_planche", "IL sans planche", True)):
        if os.path.exists(os.path.join(HERE, f"{nom}.pt")):
            f = il.fonction(il.charger(nom), sans=sans)
            yield lib, f, "brut"
            if not sans:
                yield "IL + accessible", f, "accessible"
                yield "IL + jumeau", f, "jumeau"
    if os.path.exists(os.path.join(ROOT, "rl", "decision_rl.pt")):
        import train_decision_rl as rl
        f = rl.choisir_glouton(rl.charger())
        yield "RL", f, "brut"
        yield "RL + accessible", f, "accessible"
        yield "RL + jumeau", f, "jumeau"


def figure(tous):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    noms = list(tous)
    fig, ax = plt.subplots(figsize=(13, 4.4))
    x = np.arange(len(CATS) + 1)
    w = 0.8 / len(noms)
    for i, nom in enumerate(noms):
        rs_all = tous[nom]
        taux = [100 * np.mean([r["succes"] for r in rs_all])]
        for po, ch in CATS:
            rs = [r for r in rs_all if r["posture"] == po and r["sur_chemin"] == ch]
            taux.append(100 * np.mean([r["succes"] for r in rs]) if rs else 0)
        ax.bar(x + i * w, taux, w, label=nom)
    ax.set_xticks(x + w * (len(noms) - 1) / 2)
    ax.set_xticklabels(["toutes"] + [f"{po}\n{'sur le chemin' if ch else 'ailleurs'}" for po, ch in CATS], fontsize=8)
    ax.set_ylabel("réussite (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(noms), 7), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(SORTIE + ".png", dpi=130)
    print(f"Figure : {SORTIE}.png")


def main():
    from eval_planche import resume
    if "--figure" in sys.argv:
        with open(SORTIE + ".json") as f:
            figure(json.load(f))
        return
    from decisions_planche import evaluer_decisions
    tous = {}
    ref = os.path.join(ROOT, "results", "compare_planche.json")
    if os.path.exists(ref):
        with open(ref) as f:
            anciens = json.load(f)
        if "expert (planificateur)" in anciens:
            tous["planificateur complet"] = anciens["expert (planificateur)"]
            resume(tous["planificateur complet"], "planificateur complet (repris de compare_planche.json)")
    seules = sys.argv[sys.argv.index("--seulement") + 1].split(",") if "--seulement" in sys.argv else None
    for nom, f, mode in methodes():
        if seules and not any(x in nom for x in seules):
            continue
        print(f"\nÉvaluation : {nom}")
        tous[nom] = evaluer_decisions(f, n=N, nom=nom, mode=mode)
        resume(tous[nom], nom)
    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    with open(SORTIE + ".json", "w") as f:
        json.dump(tous, f, indent=1)
    figure(tous)


if __name__ == "__main__":
    main()
