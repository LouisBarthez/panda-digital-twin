"""IL sur les DÉCISIONS : un réseau imite les choix de l'expert discret.

Entrée (23) : cube (position, lacet), cible, planche (centre, hauteur du dessus, lacet, posture),
pose de départ du bras. Sorties : 3 classifications
    orientation de pince k (5)   route d'approche (6)   route de transport (6)
Le générateur classique (common/decisions_planche.py) exécute ensuite le choix.

    python avec_obstacle/il/train_decision_il.py                      -> entraîne il/decision_il.pt, évalue sur 100 scènes
    python avec_obstacle/il/train_decision_il.py --sans-planche       -> ablation : planche cachée (decision_il_sans_planche.pt)
    python avec_obstacle/il/train_decision_il.py --eval [--view] [--sans-planche] [--mode brut|accessible|jumeau]
        --mode : comment le générateur utilise le classement des choix du réseau (voir MODES dans
        common/decisions_planche.py). Par défaut : brut (le 1er choix, tel quel).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from decisions_planche import COMBOS, MODES, N_FEAT, N_K, N_R, ROUTES, evaluer_decisions  # noqa: E402
from eval_planche import resume  # noqa: E402

SANS = "--sans-planche" in sys.argv
NOM = "decision_il_sans_planche" if SANS else "decision_il"
FEAT_PLANCHE = slice(8, 16)        # centre (2), haut, lacet (2), posture (3)
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 300
N_EVAL = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "brut"
torch.manual_seed(0)


class Decideur(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.tronc = nn.Sequential(nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.1),
                                   nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1))
        self.t_k, self.t_app, self.t_tr = nn.Linear(256, N_K), nn.Linear(256, N_R), nn.Linear(256, N_R)

    def forward(self, x):
        h = self.tronc((x - self.mean) / self.std)
        return self.t_k(h), self.t_app(h), self.t_tr(h)


def masquer(f):
    if SANS:
        f = f.copy()
        f[..., FEAT_PLANCHE] = 0.0
    return f


def charger_donnees():
    import glob
    parts = [np.load(f) for f in sorted(glob.glob("decisions_planche*.npz"))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files}
    ok = d["label"][:, 0] >= 0
    return masquer(d["feat"][ok]), d["label"][ok], len(ok)


def entrainer():
    X, Y, n_tot = charger_donnees()
    print(f"{len(X)} scènes avec une option faisable (sur {n_tot})" + ("  (planche MASQUÉE)" if SANS else ""))
    nv = max(1, len(X) // 10)
    Xv, Yv, Xt, Yt = X[:nv], Y[:nv], X[nv:], Y[nv:]
    m = Decideur(Xt.mean(0), Xt.std(0) + 1e-6)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    Xt_, Yt_ = torch.as_tensor(Xt), torch.as_tensor(Yt, dtype=torch.long)
    Xv_, Yv_ = torch.as_tensor(Xv), torch.as_tensor(Yv, dtype=torch.long)
    meilleur, etat, t0 = -1, None, time.time()
    for ep in range(1, EPOCHS + 1):
        m.train()
        perm = torch.randperm(len(Xt_))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            lk, la, lt = m(Xt_[idx])
            loss = ce(lk, Yt_[idx, 0]) + ce(la, Yt_[idx, 1]) + ce(lt, Yt_[idx, 2])
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            lk, la, lt = m(Xv_)
            acc = [(l.argmax(1) == Yv_[:, j]).float().mean().item() for j, l in enumerate((lk, la, lt))]
        score = float(np.mean(acc))
        if score > meilleur:
            meilleur, etat = score, {k: v.clone() for k, v in m.state_dict().items()}
        if ep == 1 or ep % 50 == 0:
            print(f"epoch {ep:3d}  précision validation : pince {acc[0]:.2f}, approche {acc[1]:.2f}, "
                  f"transport {acc[2]:.2f}  ({time.time() - t0:.0f} s)", flush=True)
    m.load_state_dict(etat)
    m.eval()
    torch.save(m.state_dict(), f"{NOM}.pt")
    print(f"Modèle : il/{NOM}.pt (meilleure précision moyenne en validation {meilleur:.2f})")
    return m


def charger(nom=NOM):
    sd = torch.load(os.path.join(HERE, f"{nom}.pt"))
    m = Decideur(sd["mean"].numpy(), sd["std"].numpy())
    m.load_state_dict(sd)
    m.eval()
    return m


def fonction(m, sans=SANS):
    """Classement des 180 choix par le réseau (somme des log-probabilités des 3 têtes)."""
    def choisir(feat, det, q0, env):
        f = feat.copy()
        if sans:
            f[FEAT_PLANCHE] = 0.0
        with torch.no_grad():
            lk, la, lt = (torch.log_softmax(x, 1)[0].numpy() for x in m(torch.as_tensor(f)[None]))
        score = lk[COMBOS[:, 0]] + la[COMBOS[:, 1]] + lt[COMBOS[:, 2]]
        return COMBOS[np.argsort(-score)]
    return choisir


def main():
    m = charger() if "--eval" in sys.argv else entrainer()
    assert MODE in MODES, f"--mode parmi {MODES}"
    res = evaluer_decisions(fonction(m), n=N_EVAL, nom=NOM, view="--view" in sys.argv, mode=MODE)
    resume(res, f"{NOM} (mode {MODE})")
    cpt = np.bincount([r["choix"][2] for r in res], minlength=N_R)
    print("  routes de transport choisies : " + ", ".join(f"{r} {c}" for r, c in zip(ROUTES, cpt)))
    if "--view" not in sys.argv:
        with open(f"results_{NOM}_{MODE}.json", "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
