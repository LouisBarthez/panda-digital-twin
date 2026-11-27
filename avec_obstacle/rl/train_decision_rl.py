"""RL sur les DÉCISIONS : apprendre par essai-erreur, SANS expert, quel choix fait réussir vite.

Même décision que l'IL (orientation de pince, route d'approche, route de transport) et même
générateur pour l'exécuter. Ici, personne ne dit quel est le bon choix : le robot ESSAIE un choix
dans le jumeau, reçoit une récompense (+10 - 1 par seconde si réussi, -10 s'il touche la planche,
-5 sinon), et un réseau apprend à prédire la récompense de chaque choix dans chaque scène
(bandit contextuel, apprentissage d'une fonction de valeur Q(scène, choix)).
Au début il choisit au hasard (exploration), puis de plus en plus souvent le choix qu'il croit le
meilleur (epsilon décroît de 1 à 0,1).

    python avec_obstacle/rl/train_decision_rl.py                   -> 25 tours de 200 essais, tous les cœurs
    python avec_obstacle/rl/train_decision_rl.py 40 --essais 300
    python avec_obstacle/rl/train_decision_rl.py --eval [--view] [--mode brut|accessible|jumeau]

Sorties : rl/decision_rl.pt (meilleure évaluation), rl/decision_rl_log.csv, rl/decision_rl_essais.npz
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from decisions_planche import COMBOS, MODES, N_FEAT, N_K, N_R, ROUTES  # noqa: E402

N_TOURS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 25
N_ESSAIS = int(sys.argv[sys.argv.index("--essais") + 1]) if "--essais" in sys.argv else 200
PROC = int(sys.argv[sys.argv.index("--proc") + 1]) if "--proc" in sys.argv else (os.cpu_count() or 2)
N_EVAL_RAPIDE, EVAL_TOUS = 45, 5      # évaluation rapide : seeds 0..44 (~41 scènes faisables)


def encoder(feat, combos):
    """Entrée du réseau Q : caractéristiques de la scène + choix en « one-hot » (5 + 6 + 6)."""
    n = len(combos)
    oh = np.zeros((n, N_K + 2 * N_R), np.float32)
    oh[np.arange(n), combos[:, 0]] = 1
    oh[np.arange(n), N_K + combos[:, 1]] = 1
    oh[np.arange(n), N_K + N_R + combos[:, 2]] = 1
    return np.concatenate([np.repeat(feat[None], n, 0), oh], 1)


class ReseauQ(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(N_FEAT + N_K + 2 * N_R, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        x = x.clone()
        x[:, :N_FEAT] = (x[:, :N_FEAT] - self.mean) / self.std
        return self.net(x)[:, 0]


def meilleur_choix(q, feat):
    with torch.no_grad():
        v = q(torch.as_tensor(encoder(feat, COMBOS))).numpy()
    return COMBOS[int(np.argmax(v))]


# ------------------------------------------------------------------ essais en parallèle
_env = _gen = None


def _init():
    global _env, _gen
    torch.set_num_threads(1)
    from decisions_planche import Generateur
    from twin_env import TwinPlancheEnv
    _env, _gen = TwinPlancheEnv(), Generateur()


def essai(args):
    from decisions_planche import caracteristiques, executer, recompense
    seed, poids, eps = args
    rng = np.random.default_rng(seed)
    _env.reset(seed=seed)
    det, q0 = _env.detection(), _env.q_cmd.copy()
    feat = caracteristiques(det, q0)
    if poids is None or rng.random() < eps:
        c = COMBOS[rng.integers(len(COMBOS))]                 # exploration
    else:
        q = ReseauQ(poids["mean"].numpy(), poids["std"].numpy())
        q.load_state_dict(poids)
        q.eval()
        c = meilleur_choix(q, feat)                           # exploitation
    plan, _ = _gen.plan_depuis_choix(det, q0, *c)
    res = executer(_env, plan)
    return feat, c, recompense(res), res["succes"], bool(res["collision"])


def entrainer_q(q, F, C, Rw, epochs=30):
    X = torch.as_tensor(np.concatenate([encoder(f, c[None]) for f, c in zip(F, C)]))
    Y = torch.as_tensor(np.array(Rw, np.float32))
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            idx = perm[i:i + 256]
            loss = ((q(X[idx]) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    q.eval()


def choisir_glouton(q):
    """Classement des 180 choix par valeur Q décroissante (le 1er = le choix glouton)."""
    def choisir(feat, det, q0, env):
        with torch.no_grad():
            v = q(torch.as_tensor(encoder(feat, COMBOS))).numpy()
        return COMBOS[np.argsort(-v)]
    return choisir


def charger():
    sd = torch.load(os.path.join(HERE, "decision_rl.pt"))
    q = ReseauQ(sd["mean"].numpy(), sd["std"].numpy())
    q.load_state_dict(sd)
    q.eval()
    return q


def main():
    from decisions_planche import evaluer_decisions
    from eval_planche import resume
    if "--eval" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "brut"
        assert mode in MODES, f"--mode parmi {MODES}"
        res = evaluer_decisions(choisir_glouton(charger()), n=100, nom="decision_rl", view="--view" in sys.argv,
                                mode=mode)
        resume(res, f"decision_rl (mode {mode})")
        if "--view" not in sys.argv:
            import json
            with open(f"results_decision_rl_{mode}.json", "w") as f:
                json.dump(res, f, indent=1)
        return
    F, C, Rw = [], [], []
    q, poids, best = None, None, -1
    log = open("decision_rl_log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["tour", "essais", "epsilon", "reussite_collecte", "collisions_collecte", "recompense_moyenne",
                "reussite_eval", "scenes_eval", "minutes"])
    t0 = time.time()
    with Pool(PROC, initializer=_init) as pool:
        for tour in range(1, N_TOURS + 1):
            eps = max(0.1, 1.0 - (tour - 1) / max(1, 0.6 * N_TOURS))
            seeds = range(100000 + tour * 1000, 100000 + tour * 1000 + N_ESSAIS)
            out = pool.map(essai, [(s, poids, eps) for s in seeds])
            for f, c, r, _, _ in out:
                F.append(f)
                C.append(c)
                Rw.append(r)
            ok = sum(o[3] for o in out)
            col = sum(o[4] for o in out)
            if q is None:
                Fa = np.array(F)
                q = ReseauQ(Fa.mean(0), Fa.std(0) + 1e-6)
            entrainer_q(q, F, C, Rw)
            poids = {k: v.clone() for k, v in q.state_dict().items()}
            ev, n_ev = "", ""
            if tour % EVAL_TOUS == 0 or tour == N_TOURS:
                res = evaluer_decisions(choisir_glouton(q), n=N_EVAL_RAPIDE, nom=f"RL tour {tour}")
                ev, n_ev = sum(r["succes"] for r in res), len(res)
                if ev >= best:
                    best = ev
                    torch.save(q.state_dict(), "decision_rl.pt")
            mn = (time.time() - t0) / 60
            print(f"tour {tour:2d}  epsilon {eps:.2f}  collecte {ok}/{N_ESSAIS} réussis, {col} collisions, "
                  f"récompense moyenne {np.mean([o[2] for o in out]):+.1f}"
                  + (f"  | évaluation {ev}/{n_ev}" if ev != "" else "") + f"  ({mn:.0f} min)", flush=True)
            w.writerow([tour, len(Rw), round(eps, 2), ok, col, round(float(np.mean([o[2] for o in out])), 2),
                        ev, n_ev, round(mn, 1)])
            log.flush()
            np.savez_compressed("decision_rl_essais.npz", feat=np.array(F), choix=np.array(C), recompense=np.array(Rw))
    print(f"\nMeilleur modèle : rl/decision_rl.pt  |  évaluer : python avec_obstacle/rl/train_decision_rl.py --eval")
    print("Routes de transport les plus jouées à la fin : " +
          ", ".join(f"{ROUTES[r]} {c}" for r, c in enumerate(np.bincount(np.array(C)[-N_ESSAIS:, 2], minlength=N_R))))


if __name__ == "__main__":
    main()
