"""DAgger avec le planificateur comme expert (jumeau avec planche).

Le BC seul échoue surtout parce que ses petites erreurs l'amènent dans des états que l'expert ne
visite jamais (prise décalée de quelques mm, cube un peu tourné dans la pince...), et il n'a jamais
appris quoi faire là. À chaque itération :
  1. le BC joue des épisodes (avec une probabilité beta, c'est l'action de l'expert qui est jouée :
     beta = 0,5 puis 0,25 puis 0) ;
  2. dans CHAQUE état visité, l'expert dit ce qu'il aurait fait : il suit son plan si l'état en est
     proche, sinon il REPLANIFIE depuis cet état (cube tenu ou non) -> expert_planche.etiquette() ;
  3. ces exemples s'ajoutent aux démos, le BC est ré-entraîné, puis évalué.

Il faut avoir lancé avant : record_demos_planche.py puis train_bc_planche.py.

    python avec_obstacle/il/dagger_planche.py                  -> 4 itérations de 120 épisodes, tous les cœurs
    python avec_obstacle/il/dagger_planche.py 6 --ep 200 --proc 6
Sorties : il/bc_dagger_planche.pt (meilleure itération), bc_dagger_planche_itN.pt, il/dagger_planche_log.csv,
          il/dagger_planche_data.npz (états étiquetés, réutilisés si on relance)
Évaluer / voir : python avec_obstacle/il/train_bc_planche.py --eval --modele bc_dagger_planche [--view]
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

N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
N_EP = int(sys.argv[sys.argv.index("--ep") + 1]) if "--ep" in sys.argv else 120
PROC = int(sys.argv[sys.argv.index("--proc") + 1]) if "--proc" in sys.argv else (os.cpu_count() or 2)
BETAS = [0.5, 0.25, 0.0]
EPOCHS_FT = 15
N_EVAL = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 100

_env = _ex = None


def _init():
    global _env, _ex
    torch.set_num_threads(1)
    from expert_planche import ExpertPlanche
    from twin_env import TwinPlancheEnv
    _env, _ex = TwinPlancheEnv(), ExpertPlanche()


def episode(args):
    """Le BC joue (mélangé à l'expert avec la proba beta) ; l'expert étiquette chaque état."""
    global _pol
    seed, poids, beta = args
    import train_bc_planche as bc
    from twin_env import MAX_STEPS
    torch.set_num_threads(1)                 # un cœur par processus
    pol = bc.Politique(poids["mean"].numpy(), poids["std"].numpy())
    pol.load_state_dict(poids)
    pol.eval()
    env, ex = _env, _ex
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    ex.oublier()
    O, A, info, stable = [], [], {}, 0
    for _ in range(MAX_STEPS):
        with torch.no_grad():
            a_bc = pol(torch.as_tensor(obs, dtype=torch.float32)[None])[0].numpy()
        a_ex = ex.etiquette(env)
        if a_ex is not None:
            O.append(obs.copy())
            A.append(a_ex.copy())
        a = a_ex if (a_ex is not None and rng.random() < beta) else a_bc
        obs, _, term, trunc, info = env.step(a)
        stable = stable + 1 if info["is_success"] else 0
        if term or trunc or stable >= 25:
            break
    return (np.array(O, np.float32).reshape(-1, len(obs)), np.array(A, np.float32).reshape(-1, 8),
            bool(info.get("is_success")), info.get("collision", ""))


def reentrainer(pol, obs, act, n_demo_ep_val):
    """Ré-entraîne la politique (à partir de ses poids actuels) sur démos + états DAgger."""
    import train_bc_planche as bc
    X, Y = torch.as_tensor(bc.masquer(obs)), torch.as_tensor(act)
    Xv, Yv = n_demo_ep_val
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_FT)
    meilleur, etat = float("inf"), None
    for ep in range(EPOCHS_FT):
        pol.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bc.BATCH):
            idx = perm[i:i + bc.BATCH]
            loss = ((pol(X[idx], bc.BRUIT_OBS) - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        pol.eval()
        with torch.no_grad():
            lv = ((pol(Xv) - Yv) ** 2).mean().item()
        if lv < meilleur:
            meilleur, etat = lv, {k: v.clone() for k, v in pol.state_dict().items()}
    pol.load_state_dict(etat)
    pol.eval()
    return meilleur


def main():
    import train_bc_planche as bc
    from eval_planche import evaluer, resume
    torch.set_num_threads(os.cpu_count() or 2)
    obs_d, act_d, ep_d = bc.charger_demos()
    val = np.isin(ep_d, np.arange(max(1, (int(ep_d.max()) + 1) // 10)))
    Xv, Yv = torch.as_tensor(bc.masquer(obs_d[val])), torch.as_tensor(act_d[val])
    pol = bc.charger("bc_dagger_planche" if os.path.exists("bc_dagger_planche.pt") and "--suite" in sys.argv
                     else "bc_planche")
    O_dag, A_dag = np.zeros((0, obs_d.shape[1]), np.float32), np.zeros((0, 8), np.float32)
    if "--suite" in sys.argv and os.path.exists("dagger_planche_data.npz"):
        d = np.load("dagger_planche_data.npz")
        O_dag, A_dag = d["obs"], d["act"]
    nouveau = not os.path.exists("dagger_planche_log.csv") or "--suite" not in sys.argv
    log = open("dagger_planche_log.csv", "w" if nouveau else "a", newline="")
    w = csv.writer(log)
    if nouveau:
        w.writerow(["iteration", "beta", "episodes", "reussites_pendant_collecte", "etats_etiquetes",
                    "loss_validation", "reussite_eval", "scenes_eval", "collisions_eval", "minutes"])
    meilleur_eval = -1
    with Pool(PROC, initializer=_init) as pool:
        for it in range(1, N_ITER + 1):
            t0 = time.time()
            beta = BETAS[min(it - 1, len(BETAS) - 1)]
            poids = {k: v.clone() for k, v in pol.state_dict().items()}
            seeds = range(20000 + 1000 * it, 20000 + 1000 * it + N_EP)
            print(f"\n=== Itération {it}/{N_ITER} (beta {beta}) : {N_EP} épisodes, {PROC} processus ===", flush=True)
            res = pool.map(episode, [(s, poids, beta) for s in seeds])
            O_dag = np.concatenate([O_dag] + [r[0] for r in res])
            A_dag = np.concatenate([A_dag] + [r[1] for r in res])
            ok = sum(r[2] for r in res)
            print(f"  collecte : {ok}/{N_EP} réussis, {sum(len(r[0]) for r in res)} états étiquetés "
                  f"(total DAgger {len(O_dag)})  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            np.savez_compressed("dagger_planche_data.npz", obs=O_dag, act=A_dag)
            lv = reentrainer(pol, np.concatenate([obs_d[~val], O_dag]), np.concatenate([act_d[~val], A_dag]),
                             (Xv, Yv))
            torch.save(pol.state_dict(), f"bc_dagger_planche_it{it}.pt")
            r_eval = evaluer(bc.fonction(pol), n=N_EVAL, obs_planche=not bc.SANS, nom=f"DAgger it{it}")
            ok_e = sum(r["succes"] for r in r_eval)
            if ok_e >= meilleur_eval:          # on garde la MEILLEURE itération (la 4e s'était effondrée le 24/11)
                meilleur_eval = ok_e
                torch.save(pol.state_dict(), "bc_dagger_planche.pt")
            col = sum(bool(r["collision"]) for r in r_eval)
            print(f"  ré-entraînement : validation {lv:.4f} ; évaluation {ok_e}/{len(r_eval)} réussies, "
                  f"{col} collisions  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            w.writerow([it, beta, N_EP, ok, len(O_dag), round(lv, 5), ok_e, len(r_eval), col,
                        round((time.time() - t0) / 60, 1)])
            log.flush()
    resume(r_eval, "BC + DAgger (dernière itération)")
    print("\nModèle : il/bc_dagger_planche.pt  |  voir : python avec_obstacle/il/train_bc_planche.py --eval --modele "
          "bc_dagger_planche --view")


if __name__ == "__main__":
    main()
