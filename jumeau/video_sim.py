"""Vidéo « propre » d'un épisode SIMULÉ de l'expert, pour le rapport et le GitHub.

Rejoue dans la scène twin la dernière trajectoire envoyée au robot (jumeau/derniere_trajectoire.json,
écrite par jumeau_live.py à chaque touche R), ou une configuration donnée à la main, et l'enregistre
en 1280x720 depuis une ou plusieurs caméras.

Usage (PowerShell, venv activé, dans C:\\Users\\louis\\pfe_mujoco) :
    python jumeau/video_sim.py                          -> dernière trajectoire, vue 3/4
    python jumeau/video_sim.py --vue webcam             -> vue de la webcam (même cadrage que le réel)
    python jumeau/video_sim.py --vue toutes             -> 3/4 | webcam côte à côte (2560x720)
    python jumeau/video_sim.py --cube 0.55 -0.10 30 --cible 0.60 0.15
    --gif : ajoute un GIF léger (640 px de large, 12 img/s) pour le README GitHub
Sortie : jumeau/videos/<date>_sim_<vue>.mp4 (et .gif)
"""
import datetime
import json
import os
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from planif_expert import HOME_Q, Jumeau, planifier, rejouer  # noqa: E402

W, H = 1280, 720
FPS = 25                      # le rejeu avance par 20 ms (50 img/s) : on garde 1 image sur 2


def arg(nom, n):
    i = sys.argv.index(nom)
    return [float(v) for v in sys.argv[i + 1:i + 1 + n]]


def camera_34():
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = [0.42, 0.0, 0.10]
    c.distance, c.azimuth, c.elevation = 1.25, 150, -28
    return c


def titre(img, lignes):
    for i, t in enumerate(lignes):
        cv2.putText(img, t, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (20, 40 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    j = Jumeau()
    if "--cube" in sys.argv:
        cx, cy, lacet = arg("--cube", 3)
        tx, ty = arg("--cible", 2)
        q0 = HOME_Q.copy()
        etapes, msg = planifier(j, q0, (cx, cy), lacet, (tx, ty))
        if etapes is None:
            sys.exit(f"plan impossible : {msg}")
        cube, cible = (cx, cy), (tx, ty)
    else:
        f = os.path.join(HERE, "derniere_trajectoire.json")
        if not os.path.exists(f):
            sys.exit("jumeau/derniere_trajectoire.json absent : faire un essai avec R, ou donner --cube et --cible")
        with open(f) as fp:
            traj = json.load(fp)
        info = traj["info"]
        cube, lacet, cible = info["cube"], info["lacet"], info["cible"]
        q0 = np.array(traj["q_depart"])
        etapes = traj["etapes"]

    vue = sys.argv[sys.argv.index("--vue") + 1] if "--vue" in sys.argv else "3-4"
    vues = ["3-4", "webcam"] if vue == "toutes" else [vue]
    rend = mujoco.Renderer(j.model, H, W)
    c34 = camera_34()

    os.makedirs(os.path.join(HERE, "videos"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    chemin = os.path.join(HERE, "videos", f"{stamp}_sim_{vue}.mp4")
    w = imageio.get_writer(chemin, fps=FPS, codec="libx264", quality=8, macro_block_size=16,
                           ffmpeg_params=["-preset", "veryfast"])
    gif = [] if "--gif" in sys.argv else None

    g = rejouer(j, q0, cube, lacet, cible, etapes)
    k, bilan = 0, None
    lignes = ["Jumeau numerique - expert simule (MuJoCo)",
              f"cube ({cube[0]:.3f} ; {cube[1]:+.3f}) m, lacet {lacet:.0f} deg -> cible ({cible[0]:.3f} ; {cible[1]:+.3f}) m"]
    while True:
        try:
            next(g)
        except StopIteration as fin:
            bilan = fin.value
            break
        k += 1
        if k % 2:
            continue
        imgs = []
        for v in vues:
            rend.update_scene(j.data, camera=c34 if v == "3-4" else v)
            imgs.append(rend.render().copy())
        img = np.hstack(imgs)
        img = titre(img[:, :, ::-1].copy(), lignes)[:, :, ::-1]
        w.append_data(img)
        if gif is not None and k % 4 == 0:
            gif.append(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
    # 2 s sur l'image finale avec le bilan
    fin_img = titre(img[:, :, ::-1].copy(), lignes + [("REUSSI : " if bilan["ok"] else "ECHEC : ") + bilan["message"]])[:, :, ::-1]
    for _ in range(2 * FPS):
        w.append_data(fin_img)
    w.close()
    print(f"{chemin}  ({bilan['message']})")
    if gif is not None:
        gif += [cv2.resize(fin_img, gif[0].shape[1::-1])] * 24
        chemin_gif = chemin[:-4] + ".gif"
        imageio.mimsave(chemin_gif, gif, duration=1 / 12, loop=0)
        print(f"{chemin_gif}  ({os.path.getsize(chemin_gif) / 1e6:.1f} Mo)")
    rend.close()


if __name__ == "__main__":
    main()
