"""Behavior cloning dans le jumeau avec planche : un MLP imite l'expert planificateur.

Observation (40) : angles, vitesses, pince, cube (position + lacet relatif à la pince), cible, et les
8 valeurs de la planche (centre, hauteur du dessus, lacet, posture). Action (8) : 7 petits déplacements
articulaires + pince. Le réseau ne planifie rien : il doit avoir APPRIS où passer selon la planche.

    python avec_obstacle/il/train_bc_planche.py                     -> entraîne bc_planche.pt, évalue sur 100 scènes
    python avec_obstacle/il/train_bc_planche.py --sans-planche      -> ablation : la planche est cachée au réseau
                                                         (bc_sans_planche.pt). Si le score ne baisse pas,
                                                         la tâche ne force pas à regarder la planche.
    python avec_obstacle/il/train_bc_planche.py --eval [--sans-planche] [--view] [--n 200]
    python avec_obstacle/il/train_bc_planche.py --eval --modele bc_dagger_planche [--view]   -> modèle issu de DAgger
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

from eval_planche import evaluer, resume  # noqa: E402
from twin_env import OBS_DIM, OBS_PLANCHE  # noqa: E402

SANS = "--sans-planche" in sys.argv
NOM = "bc_sans_planche" if SANS else "bc_planche"
MODELE = sys.argv[sys.argv.index("--modele") + 1] if "--modele" in sys.argv else NOM
EPOCHS = int(sys.argv[sys.argv.index("--epochs") + 1]) if "--epochs" in sys.argv else 80
N_EVAL = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100
BATCH, LR, HIDDEN = 512, 1e-3, 512
DROPOUT = 0.1
BRUIT_OBS = 0.05       # bruit gaussien sur les observations normalisées pendant l'entraînement
                       # (le BC surapprenait : perte de validation qui remonte dès l'epoch 10)
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 4)


class Politique(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 8), nn.Tanh())

    def forward(self, obs, bruit=0.0):
        x = (obs - self.mean) / self.std
        if bruit > 0:
            x = x + bruit * torch.randn_like(x)
        return self.net(x)


def masquer(obs):
    if SANS:
        obs = obs.copy()
        obs[:, OBS_PLANCHE] = 0.0
    return obs


def charger_demos():
    """Toutes les démos : demos_planche.npz + demos_planche_XXXX.npz (autres seeds)."""
    import glob
    O, A, E, n0 = [], [], [], 0
    for fich in sorted(glob.glob("demos_planche*.npz")):
        d = np.load(fich)
        O.append(d["obs"])
        A.append(d["act"])
        E.append(d["episode"] + n0)
        n0 += int(d["episode"].max()) + 1
        print(f"  {fich} : {int(d['episode'].max()) + 1} épisodes")
    return np.concatenate(O), np.concatenate(A), np.concatenate(E)


def entrainer():
    obs, act, ep = charger_demos()
    obs = masquer(obs)
    n_ep = int(ep.max()) + 1
    print(f"{n_ep} épisodes, {len(obs)} pas" + ("  (planche MASQUÉE)" if SANS else ""))
    val = np.isin(ep, np.arange(max(1, n_ep // 10)))
    mean, std = obs[~val].mean(0), obs[~val].std(0) + 1e-6
    X, Y = torch.as_tensor(obs[~val]), torch.as_tensor(act[~val])
    Xv, Yv = torch.as_tensor(obs[val]), torch.as_tensor(act[val])
    pol = Politique(mean, std)
    opt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=1e-4)
    meilleur, etat_meilleur = float("inf"), None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        pol.train()
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            loss = ((pol(X[idx], BRUIT_OBS) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        pol.eval()
        with torch.no_grad():
            lv = ((pol(Xv) - Yv) ** 2).mean().item()
        if lv < meilleur:                      # on garde le modèle qui généralise le mieux
            meilleur = lv
            etat_meilleur = {k: v.clone() for k, v in pol.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch {epoch:3d}  loss entraînement {tot / len(X):.4f}  validation {lv:.4f}"
                  f"  (meilleure {meilleur:.4f}, {time.time() - t0:.0f} s)", flush=True)
    pol.load_state_dict(etat_meilleur)
    pol.eval()
    torch.save(pol.state_dict(), f"{NOM}.pt")
    print(f"Modèle : il/{NOM}.pt")
    return pol


def charger(nom=NOM):
    sd = torch.load(os.path.join(HERE, f"{nom}.pt"))
    pol = Politique(sd["mean"].numpy(), sd["std"].numpy())
    pol.load_state_dict(sd)
    pol.eval()
    return pol


def fonction(pol):
    def f(env, obs):
        with torch.no_grad():
            return pol(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
    return f


def main():
    pol = charger(MODELE) if "--eval" in sys.argv else entrainer()
    res = evaluer(fonction(pol), n=N_EVAL, obs_planche=not SANS, view="--view" in sys.argv, nom=MODELE)
    resume(res, MODELE)
    if "--view" not in sys.argv:
        with open(f"results_{MODELE}.json", "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
