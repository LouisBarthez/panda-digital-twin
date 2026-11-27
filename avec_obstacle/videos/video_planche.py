"""Vidéo d'épisodes dans le jumeau avec planche : vue 3/4 | vue de la webcam, légendée.

    python avec_obstacle/videos/video_planche.py                                  -> expert, 3 scènes types
    python avec_obstacle/videos/video_planche.py --seeds 14 1 17 --politique bc_dagger_planche
    python avec_obstacle/videos/video_planche.py --gif                             -> GIF léger en plus (README)
    python avec_obstacle/videos/video_planche.py --politique decision_il --mode jumeau   -> IL sur les décisions
    python avec_obstacle/videos/video_planche.py --politique decision_rl --mode brut     -> RL sur les décisions
    (--mode : brut | accessible | jumeau, voir common/decisions_planche.py)

Scènes types (seeds d'évaluation) : 14 = planche couchée sur le chemin (passage au ras),
1 = planche sur la tranche (passage par-dessus), 17 = planche debout (contournement).
Sortie : videos/planche_<politique>.mp4 (+ .gif)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "il"))
sys.path.insert(0, os.path.join(ROOT, "rl"))

import cv2  # noqa: E402
import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from eval_planche import PAS_STABLE, scenes_eval  # noqa: E402
from twin_env import MAX_STEPS, TwinPlancheEnv  # noqa: E402

W, H = 640, 400
POSTURE = {"plat": "couchee (3 cm)", "tranche": "sur la tranche (10 cm)", "debout": "debout (54 cm)"}


def arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


def politique(nom):
    if nom.startswith("decision"):
        from decisions_planche import Generateur, SuiveurPlan, caracteristiques, decider
        if nom.startswith("decision_rl"):
            import train_decision_rl as rl
            choisir = rl.choisir_glouton(rl.charger())
        else:
            import train_decision_il as il
            choisir = il.fonction(il.charger(nom), sans=nom.endswith("sans_planche"))
        gen, mode, etat = Generateur(), arg("--mode", "brut"), {}

        def reset(env):
            det, q0 = env.detection(), env.q_cmd.copy()
            plan, msg, _, _ = decider(gen, choisir(caracteristiques(det, q0), det, q0, env), det, q0, mode)
            etat["s"] = SuiveurPlan(plan)
            etat["msg"] = msg
            print(f"  décision : {msg}")
        return (lambda env, obs: etat["s"].action(env)), reset
    if nom == "expert":
        from expert_planche import ExpertPlanche
        ex = ExpertPlanche()
        return (lambda env, obs: ex.action(env)), ex.reset
    import train_bc_planche as bc
    return bc.fonction(bc.charger(nom)), None


def texte(img, lignes, couleur=(255, 255, 255)):
    for i, t in enumerate(lignes):
        y = 24 + 24 * i
        cv2.rectangle(img, (6, y - 18), (12 + 10 * len(t), y + 6), (0, 0, 0), -1)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, couleur, 1, cv2.LINE_AA)
    return img


def main():
    nom = arg("--politique", "expert")
    seeds = [int(s) for s in sys.argv[sys.argv.index("--seeds") + 1:] if s.isdigit()] if "--seeds" in sys.argv \
        else [14, 1, 17]
    scenes = dict(scenes_eval(max(seeds) + 1))
    f, cb = politique(nom)
    env = TwinPlancheEnv()
    r = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.5, 0.05, 0.2]
    cam.distance, cam.azimuth, cam.elevation = 1.55, 205, -22
    os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
    suffixe = f"_{arg('--mode', 'brut')}" if nom.startswith("decision") else ""
    sortie = os.path.join(ROOT, "videos", f"planche_{nom}{suffixe}.mp4")
    images = []
    for s in seeds:
        if s not in scenes:
            print(f"seed {s} : scène sans plan possible, sautée")
            continue
        sc = scenes[s]
        obs, _ = env.reset(seed=s, options={"scene": sc["scene"], "q0": sc["q0"]})
        if cb:
            cb(env)
        lib = {"decision_il": "IL", "decision_rl": "RL"}.get(nom, nom)
        if nom.startswith("decision"):
            lib += f" ({arg('--mode', 'brut')})"
        titre = f"{lib} - planche {POSTURE[sc['scene']['posture']]}"
        stable, info = 0, {}
        for t in range(MAX_STEPS):
            obs, _, term, trunc, info = env.step(f(env, obs))
            if t % 2 == 0:
                r.update_scene(env.data, camera=cam)
                a = r.render().copy()
                r.update_scene(env.data, camera="webcam")
                b = r.render().copy()
                img = np.concatenate([texte(a, [titre, f"t = {info['pas'] * 0.04:4.1f} s"]),
                                      texte(b, ["vue de la webcam (jumeau)"])], axis=1)
                images.append(img)
            stable = stable + 1 if info["is_success"] else 0
            if term or trunc or stable >= PAS_STABLE:
                break
        fin = "REUSSI" if info["is_success"] else f"ECHEC {info['collision']}"
        coul = (120, 255, 120) if info["is_success"] else (255, 120, 120)
        derniere = images[-1].copy()
        texte(derniere, [titre, fin], coul)
        images += [derniere] * 15
        print(f"seed {s} ({sc['scene']['posture']}) : {fin}")
    imageio.mimsave(sortie, images, fps=12, macro_block_size=8)
    print(f"Vidéo : {sortie}")
    if "--gif" in sys.argv:
        petites = [cv2.resize(i, (i.shape[1] // 2, i.shape[0] // 2)) for i in images[::2]]
        imageio.mimsave(sortie.replace(".mp4", ".gif"), petites, duration=1 / 6, loop=0)
        print(f"GIF : {sortie.replace('.mp4', '.gif')}")


if __name__ == "__main__":
    main()
