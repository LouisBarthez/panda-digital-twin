"""Superpose la vue de la webcam VIRTUELLE (scène twin MuJoCo) sur l'image RÉELLE de la webcam.
Si la table, le tag et le robot tombent au même endroit, la scène twin est géométriquement fidèle.

Usage (PowerShell, dans C:\\Users\\louis\\pfe_mujoco) :
    python perception/superposition.py                         -> image live de la webcam (caméra 1)
    python perception/superposition.py --image perception/vue_webcam.png
    python perception/superposition.py --q "0 -0.3 0 -2.2 0 1.9 0.8"   -> robot simulé dans cette pose
    (cube et cible : placés aux positions de perception/detection.json s'il existe ; --sans-detection sinon)
Sorties (dans perception/) :
    superposition.png        image réelle corrigée + contours de la sim en jaune + coins du tag
    superposition_mix.png    mélange 50 % réel / 50 % sim
    reel_corrige.png         image réelle remappée sur la caméra idéale (sans distorsion)

Convention (SUIVI_PFE §2) : la sim rend un sténopé idéal centré (fovy réel), donc l'image réelle est
d'abord remappée sur cette caméra idéale avec cv2.initUndistortRectifyMap(K, D, None, K_ideal).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, HERE)
from webcam_index import index_webcam  # noqa: E402

from twin_scene import HOME_Q, WEBCAM, build_twin_model, render_webcam, set_arm  # noqa: E402

import mujoco  # noqa: E402

W, H = 1280, 720


def load_intrinsics():
    """Même lecture que apriltag_pose.py (clés K / camera_matrix / fx..., distorsion '*dist*')."""
    from apriltag_pose import load_calibration
    K, D, _ = load_calibration(os.path.join(HERE, "webcam_intrinsics.json"))
    return K, D


def ideal_K():
    f = (H / 2) / np.tan(np.radians(WEBCAM["fovy"]) / 2)
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])


def grab_live(index=1):
    cap = cv2.VideoCapture(index_webcam(), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    img = None
    for _ in range(40):
        ok, img = cap.read()
    cap.release()
    if img is None or img.shape[:2] != (H, W):
        raise SystemExit("[ERREUR] pas d'image 1280x720 sur la caméra 1 (appli Caméra fermée ?)")
    return img


def main():
    args = sys.argv
    q = HOME_Q
    if "--q" in args:
        q = np.array([float(v) for v in args[args.index("--q") + 1].split()])
    real = cv2.imread(args[args.index("--image") + 1]) if "--image" in args else grab_live()
    if real is None or real.shape[:2] != (H, W):
        raise SystemExit("[ERREUR] l'image réelle doit faire 1280x720")

    K, D = load_intrinsics()
    Ki = ideal_K()
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Ki, (W, H), cv2.CV_32FC1)
    real_u = cv2.remap(real, m1, m2, cv2.INTER_LINEAR)

    # cube et cible placés aux positions détectées sur l'image réelle (detection.py), si disponibles
    kw = {}
    det_path = os.path.join(HERE, "detection.json")
    if "--sans-detection" not in args and os.path.exists(det_path):
        with open(det_path) as f:
            det = json.load(f)
        if det.get("cube"):
            kw["cube_xy"] = det["cube"]["xyz"][:2]
            kw["cube_yaw"] = np.radians(det["cube"].get("lacet_deg", 0.0))
        if det.get("cible"):
            kw["target_xy"] = det["cible"]["xyz"][:2]
        print(f"Cube et cible placés aux positions détectées ({det_path})")
    model = build_twin_model(**kw)
    data = mujoco.MjData(model)
    set_arm(model, data, q)
    sim = render_webcam(model, data)[:, :, ::-1].copy()       # RGB -> BGR

    # contours de la sim (jaune) : frontières entre objets, à partir de l'image de segmentation
    # (id de géométrie par pixel), plus nets qu'un Canny sur l'image couleur (ombres, lumière)
    seg = render_webcam(model, data, segmentation=True).astype(np.int32)
    edges = np.zeros(seg.shape, np.uint8)
    edges[:, 1:][seg[:, 1:] != seg[:, :-1]] = 255
    edges[1:, :][seg[1:, :] != seg[:-1, :]] = 255
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    over = real_u.copy()
    over[edges > 0] = (0, 255, 255)

    # coins du tag : projetés depuis la sim (jaune) et détectés dans l'image réelle (magenta)
    cam = model.camera("webcam").id
    Rc = data.cam_xmat[cam].reshape(3, 3)
    pc = data.cam_xpos[cam]
    for i in range(4):
        p = data.site_xpos[model.site(f"tag_c{i}").id]
        v = Rc.T @ (p - pc)                                   # repère caméra MuJoCo (regarde vers -z)
        u = Ki[0, 0] * v[0] / -v[2] + Ki[0, 2]
        w_ = -Ki[1, 1] * v[1] / -v[2] + Ki[1, 2]
        cv2.circle(over, (int(round(u)), int(round(w_))), 5, (0, 255, 255), -1)
    try:
        dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        if hasattr(cv2.aruco, "ArucoDetector"):
            corners, ids, _ = cv2.aruco.ArucoDetector(dct, cv2.aruco.DetectorParameters()).detectMarkers(
                cv2.cvtColor(real_u, cv2.COLOR_BGR2GRAY))
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(cv2.cvtColor(real_u, cv2.COLOR_BGR2GRAY), dct)
        if ids is not None and 0 in ids.flatten():
            c = corners[list(ids.flatten()).index(0)][0]
            for p in c:
                cv2.circle(over, (int(p[0]), int(p[1])), 7, (255, 0, 255), 2)
    except Exception as e:  # noqa: BLE001
        print(f"(détection du tag ignorée : {e})")

    cv2.putText(over, "jaune = sim (twin) | magenta = tag detecte dans l'image reelle", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mix = cv2.addWeighted(real_u, 0.5, sim, 0.5, 0)

    # côte à côte : réel (corrigé de la distorsion) | virtuel, même point de vue, avec légendes
    def legende(img, txt):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (len(txt) * 17 + 20, 44), (0, 0, 0), -1)
        cv2.putText(img, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img
    sep = np.full((H, 8, 3), 255, np.uint8)
    cote = np.hstack([legende(real_u, "Reel (webcam)"), sep, legende(sim, "Virtuel (scene twin MuJoCo)")])

    sorties = (("superposition.png", over), ("superposition_mix.png", mix), ("cote_a_cote.png", cote),
               ("reel_corrige.png", real_u), ("sim_webcam.png", sim))
    for name, img in sorties:
        cv2.imwrite(os.path.join(HERE, name), img)
    print("Écrit dans perception/ : superposition.png, superposition_mix.png, cote_a_cote.png, "
          "reel_corrige.png, sim_webcam.png")

    # copie datée des figures utiles au rapport dans rapport/figures/
    import datetime
    fig = os.path.join(HERE, "..", "rapport", "figures")
    os.makedirs(fig, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    for name, img in sorties[:3]:
        cv2.imwrite(os.path.join(fig, f"{stamp}_{name}"), img)
    print(f"Copie pour le rapport : rapport/figures/{stamp}_*.png")
    cv2.imshow("Superposition sim / reel (touche pour fermer)", over)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
