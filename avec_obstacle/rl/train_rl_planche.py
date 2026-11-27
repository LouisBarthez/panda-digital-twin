"""RL (SAC) dans le jumeau avec planche.

Même observation et même action que le BC (env common/twin_env.py) : la comparaison IL / RL est
enfin équitable (avant : BC sur images contre SAC sur état exact).

    python avec_obstacle/rl/train_rl_planche.py 1000000             -> SAC seul, à partir de rien
    python avec_obstacle/rl/train_rl_planche.py 1000000 --demos     -> SAC + démos de l'expert versées dans sa mémoire
                                                         au départ (il/demos_planche.npz) : il part de
                                                         l'exemple de l'expert, puis peut faire mieux
    python avec_obstacle/rl/train_rl_planche.py --resume [--demos]  -> reprend le dernier entraînement
    python avec_obstacle/rl/train_rl_planche.py --eval [--demos] [--view]

Évaluation tous les 25 000 pas sur 20 scènes fixes (les mêmes que l'expert et le BC) ->
rl/rl_planche[_demos]_log.csv ; meilleur modèle : rl/sac_planche[_demos]_best.zip.
Récompense (twin_env.py) : étagée approche < prise < transport < réussite, -10 et fin d'épisode
si la planche est touchée, petit coût par radian parcouru (le chemin court est récompensé).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
os.chdir(HERE)

import numpy as np  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from eval_planche import evaluer, jouer, resume, scenes_eval  # noqa: E402
from twin_env import TwinPlancheEnv  # noqa: E402

DEMOS = "--demos" in sys.argv
NOM = "sac_planche_demos" if DEMOS else "sac_planche"
TOTAL = next((int(a) for a in sys.argv[1:] if a.isdigit()), 1_000_000)
EVAL_EVERY, N_EVAL_RAPIDE = 25_000, 20


def make_env():
    return Monitor(TwinPlancheEnv(), info_keywords=("is_success",))


def politique(model, venv):
    def f(env, obs):
        a, _ = model.predict(venv.normalize_obs(obs[None].astype(np.float32)), deterministic=True)
        return a[0]
    return f


class Suivi(BaseCallback):
    def __init__(self, venv):
        super().__init__()
        self.venv, self.best, self.t0 = venv, -1, time.time()
        self.scenes = scenes_eval(60)[:N_EVAL_RAPIDE]
        self.env_eval = TwinPlancheEnv()
        nouveau = "--resume" not in sys.argv or not os.path.exists(f"rl_{NOM.replace('sac_', '')}_log.csv")
        self.log = open(f"rl_{NOM.replace('sac_', '')}_log.csv", "w" if nouveau else "a")
        if nouveau:
            self.log.write("pas,reussites,sur,collisions,minutes\n")

    def _on_step(self):
        if self.num_timesteps % EVAL_EVERY == 0:
            f = politique(self.model, self.venv)
            res = [jouer(self.env_eval, f, s, sc) for s, sc in self.scenes]
            ok = sum(r["succes"] for r in res)
            col = sum(bool(r["collision"]) for r in res)
            mn = (time.time() - self.t0) / 60
            print(f"pas {self.num_timesteps:8d}   réussite {ok:2d}/{len(res)}   collisions {col:2d}   ({mn:.0f} min)",
                  flush=True)
            self.log.write(f"{self.num_timesteps},{ok},{len(res)},{col},{mn:.1f}\n")
            self.log.flush()
            self.model.save(f"{NOM}_last")
            self.venv.save(f"vecnorm_{NOM}_last.pkl")
            if ok >= self.best:
                self.best = ok
                self.model.save(f"{NOM}_best")
                self.venv.save(f"vecnorm_{NOM}_best.pkl")
        return True


def verser_demos(model, venv):
    """Met les transitions de l'expert dans la mémoire du SAC (et cale la normalisation dessus)."""
    import glob
    n = 0
    rb = model.replay_buffer
    for chemin in sorted(glob.glob(os.path.join(HERE, "..", "il", "demos_planche*.npz"))):
        d = np.load(chemin)
        venv.obs_rms.update(d["obs"].astype(np.float64))
        for o, o2, a, r, dn in zip(d["obs"], d["next_obs"], d["act"], d["rew"], d["done"]):
            rb.add(o[None], o2[None], a[None], np.array([r]), np.array([dn]), [{}])
        n += len(d["obs"])
    print(f"{n} transitions de l'expert versées dans la mémoire du SAC")


def main():
    if "--eval" in sys.argv:
        venv = VecNormalize.load(f"vecnorm_{NOM}_best.pkl", DummyVecEnv([make_env]))
        venv.training = False
        model = SAC.load(f"{NOM}_best")
        res = evaluer(politique(model, venv), n=100, view="--view" in sys.argv, nom=NOM)
        resume(res, NOM)
        if "--view" not in sys.argv:
            import json
            with open(f"results_{NOM}.json", "w") as f:
                json.dump(res, f, indent=1)
        return
    if "--resume" in sys.argv and os.path.exists(f"{NOM}_last.zip"):
        venv = VecNormalize.load(f"vecnorm_{NOM}_last.pkl", DummyVecEnv([make_env]))
        model = SAC.load(f"{NOM}_last", env=venv)
        print(f"Reprise de {NOM}_last.zip (la mémoire repart vide)")
    else:
        venv = VecNormalize(DummyVecEnv([make_env]), norm_obs=True, norm_reward=False)
        model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=600_000, batch_size=256,
                    gamma=0.99, learning_starts=1_000 if DEMOS else 10_000,
                    policy_kwargs=dict(net_arch=[256, 256, 256]), verbose=0, seed=0)
        if DEMOS:
            verser_demos(model, venv)
    print(f"Entraînement {NOM} : {TOTAL} pas, évaluation tous les {EVAL_EVERY} pas")
    model.learn(total_timesteps=TOTAL, callback=Suivi(venv), reset_num_timesteps="--resume" not in sys.argv)
    model.save(f"{NOM}_last")
    venv.save(f"vecnorm_{NOM}_last.pkl")
    print(f"Fini. Meilleur modèle : rl/{NOM}_best.zip  |  évaluer : python avec_obstacle/rl/train_rl_planche.py --eval"
          + (" --demos" if DEMOS else ""))


if __name__ == "__main__":
    main()
