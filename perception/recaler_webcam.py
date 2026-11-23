"""Recale la webcam globale grâce au tag 0 (à lancer en début de séance, ou si le trépied a bougé).

1. mesure la pose du tag 0 vue par la webcam (moyenne de 60 images, intrinsèques + distorsion) ;
2. la compare à la référence précédente (de combien la caméra a bougé) ;
3. met à jour webcam_tag0_ref.json et webcam_pose_base.json (pose de la webcam dans le repère
   robot : T_base_cam = T_base_tag0 . T_cam_tag0^-1), après avoir gardé une copie datée des anciens.
La scène twin (twin_scene.py) et superposition.py relisent automatiquement le nouveau fichier.

Usage (PowerShell, à la racine du dépôt) :
    python perception/recaler_webcam.py              -> mesure, compare, demande avant d'écraser
    python perception/recaler_webcam.py --oui        -> écrase sans demander
"""
import datetime
import json
import os
import shutil
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webcam_index import index_webcam  # noqa: E402
from apriltag_pose import (estimate_tag_pose, load_calibration, make_detector,  # noqa: E402
                           mean_pose, rot_angle_deg)

TAG_SIZE = 0.080
N = 60
W, H = 1280, 720


def mesurer(K, D):
    cap = cv2.VideoCapture(index_webcam(), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    detect = make_detector(cv2.aruco.DICT_APRILTAG_36h11)
    poses, essais = [], 0
    for _ in range(20):                                    # exposition auto
        cap.read()
    while len(poses) < N and essais < 4 * N:
        essais += 1
        ok, img = cap.read()
        if not ok or img.shape[:2] != (H, W):
            cap.release()
            raise SystemExit("[ERREUR] pas d'image 1280x720 sur la caméra 1 (appli Caméra fermée ?)")
        corners, ids, _ = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if ids is None or 0 not in ids.flatten():
            continue
        rvec, tvec = estimate_tag_pose(corners[list(ids.flatten()).index(0)], TAG_SIZE, K, D)
        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, tvec.flatten()
            poses.append(T)
    cap.release()
    if len(poses) < N // 2:
        raise SystemExit(f"[ERREUR] tag 0 vu sur seulement {len(poses)} images : est-il visible et dégagé ?")
    P = np.array([np.linalg.inv(T)[:3, 3] for T in poses])
    return mean_pose(poses), 1000 * np.linalg.norm(P.std(0)), img


def main():
    K, D, _ = load_calibration(os.path.join(HERE, "webcam_intrinsics.json"))
    T_ct, bruit, img = mesurer(K, D)
    print(f"Tag 0 mesuré sur {N} images, bruit de position de la caméra : {bruit:.1f} mm")

    ref_path = os.path.join(HERE, "webcam_tag0_ref.json")
    if os.path.exists(ref_path):
        T_old = np.array(json.load(open(ref_path))["T_cam_tag0"])
        C_new, C_old = np.linalg.inv(T_ct), np.linalg.inv(T_old)
        dpos = 1000 * np.linalg.norm(C_new[:3, 3] - C_old[:3, 3])
        dang = rot_angle_deg(C_old[:3, :3].T @ C_new[:3, :3])
        print(f"Écart avec l'ancienne référence : {dpos:.1f} mm, {dang:.2f}°"
              + ("  -> la webcam a BOUGÉ" if dpos > 10 or dang > 1 else "  -> pas de mouvement significatif"))

    if "--oui" not in sys.argv:
        if input("Mettre à jour la calibration de la webcam ? (o/n) ").strip().lower() != "o":
            print("Rien n'a été modifié.")
            return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    arch = os.path.join(HERE, "archives_calib")
    os.makedirs(arch, exist_ok=True)
    for name in ("webcam_tag0_ref.json", "webcam_pose_base.json"):
        if os.path.exists(os.path.join(HERE, name)):
            shutil.copy(os.path.join(HERE, name), os.path.join(arch, f"{stamp}_{name}"))

    with open(ref_path, "w") as f:
        json.dump({"T_cam_tag0": T_ct.tolist(), "tag_size": TAG_SIZE, "family": "tag36h11",
                   "res": f"{W}x{H}", "n": N, "bruit_mm": round(bruit, 2)}, f, indent=2)

    T_bt = np.array(json.load(open(os.path.join(HERE, "T_base_tag0.json")))["T_base_tag0"])
    T_bc = T_bt @ np.linalg.inv(T_ct)
    R = T_bc[:3, :3]
    fovy = float(np.degrees(2 * np.arctan(H / 2 / K[1, 1])))
    with open(os.path.join(HERE, "webcam_pose_base.json"), "w") as f:
        json.dump({"T_base_cam": T_bc.round(6).tolist(),
                   "mujoco": {"pos": T_bc[:3, 3].round(4).tolist(),
                              "xyaxes": np.r_[R[:, 0], -R[:, 1]].round(4).tolist(),
                              "fovy": round(fovy, 2), "resolution": [W, H]},
                   "chaine": "T_base_cam = T_base_tag0 @ inv(T_cam_tag0)", "date": stamp}, f, indent=2)
    cv2.imwrite(os.path.join(HERE, "vue_webcam.png"), img)
    print(f"Webcam dans le repère robot : {np.round(T_bc[:3, 3], 4)} m, "
          f"axe optique incliné de {np.degrees(np.arccos(-R[2, 2])):.1f}° / verticale")
    print(f"Fichiers mis à jour (anciens copiés dans perception/archives_calib/{stamp}_*).")
    print("Vérification : python perception/superposition.py")


if __name__ == "__main__":
    main()
