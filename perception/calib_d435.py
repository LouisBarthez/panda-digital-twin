"""Calibration main-oeil de la D435 au poignet (calcul, PC Windows).

Entrées (dans perception/) :
    poses_d435.txt   : sortie de lire_pose --libre sur l'Ubuntu (copier le fichier ou son contenu ici)
    d435_tags.json   : sortie de d435_tag.py (tag vu par la D435, mêmes étiquettes)
    T_base_tag0.json : tag 0 dans le repère robot (calage au crayon du 20/11)

Principe : le tag étant déjà connu dans le repère robot, chaque pose donne directement
    T_EE_cam = (T_base_EE)^-1 . T_base_tag0 . (T_cam_tag0)^-1
On moyenne sur toutes les poses ; l'écart entre poses donne l'incertitude. Puis méthode robuste
retenue : un seul solvePnP sur les coins du tag de toutes les poses (erreur minimisée en pixels).

Usage :
    python perception/calib_d435.py
Sorties :
    perception/d435_pose_hand.json   (lu automatiquement par common/twin_scene.py)
    perception/d435_calib/overlay_<etiquette>.png   (sim D435 calibrée superposée à l'image réelle)
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from apriltag_pose import mean_pose, rot_angle_deg  # noqa: E402

EE_OFFSET = 0.1034   # bride -> repère EE de Desk (Franka Hand) : 10,34 cm sur z, même orientation que "hand"


def lire_poses(path):
    poses = {}
    for line in open(path):
        a = line.split()
        if len(a) < 13:
            continue
        T = np.eye(4)
        T[:3, 3] = [float(v) for v in a[1:4]]
        T[:3, :3] = np.array([float(v) for v in a[4:13]]).reshape(3, 3)
        q = np.array([float(v) for v in a[13:20]]) if len(a) >= 20 else None
        poses[a[0]] = (T, q)
    return poses


def coins_tag_base(T_bt, tag_size):
    """Coins c0..c3 du carré noir dans le repère robot (ordre OpenCV)."""
    h = tag_size / 2
    C = np.array([[-h, h, 0, 1], [h, h, 0, 1], [h, -h, 0, 1], [-h, -h, 0, 1]]).T
    return (T_bt @ C)[:3].T


def detecter_coins(label):
    """Re-détecte les 4 coins du tag 0 (pixels) sur l'image enregistrée par d435_tag.py."""
    path = os.path.join(HERE, "d435_calib", f"{label}.png")
    if not os.path.exists(path):
        return None
    from apriltag_pose import make_detector
    gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    corners, ids, _ = make_detector(cv2.aruco.DICT_APRILTAG_36h11)(gray)
    if ids is None or 0 not in ids.flatten():
        return None
    c = corners[list(ids.flatten()).index(0)].reshape(4, 2).astype(np.float64)
    # affinage sous-pixel
    c32 = c.astype(np.float32).reshape(-1, 1, 2)
    cv2.cornerSubPix(gray, c32, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    return c32.reshape(4, 2).astype(np.float64)


def pnp_global(labels, poses, tags, T_bt):
    """Méthode robuste : les 4 coins du tag de TOUTES les poses, exprimés dans le repère EE
    (P_EE = T_base_EE^-1 . T_base_tag . coin), sont vus par une caméra FIXE par rapport à l'EE.
    -> un seul solvePnP sur ~48 points donne T_cam_EE, en minimisant l'erreur en PIXELS.
    (La 1re méthode multiplie des orientations de tag bruitées : à 0,8 m, 2° d'erreur sur
    l'orientation du tag font déjà ~3 cm d'erreur sur la position de la caméra.)"""
    obj, img, par_pose = [], [], {}
    K = np.array(tags[labels[0]]["K"])
    D = np.array(tags[labels[0]]["D"])
    size = tags[labels[0]].get("tag_size", 0.08)
    Cb = coins_tag_base(T_bt, size)
    for k in labels:
        c = detecter_coins(k)
        if c is None:
            print(f"  ({k} : tag non retrouvé sur l'image, pose ignorée)")
            continue
        T_eb = np.linalg.inv(poses[k][0])
        P = (T_eb[:3, :3] @ Cb.T).T + T_eb[:3, 3]
        par_pose[k] = (len(obj), len(obj) + 4)
        obj += list(P)
        img += list(c)
    obj, img = np.array(obj), np.array(img)
    if len(obj) < 12:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    # 2e passe en écartant les poses aberrantes (erreur > 3x la médiane)
    proj = cv2.projectPoints(obj, rvec, tvec, K, D)[0].reshape(-1, 2)
    err = np.linalg.norm(proj - img, axis=1)
    e_pose = {k: float(err[a:b].mean()) for k, (a, b) in par_pose.items()}
    med = np.median(list(e_pose.values()))
    garde = [k for k in e_pose if e_pose[k] <= max(3 * med, 2.0)]
    if len(garde) < len(e_pose):
        sel = np.concatenate([np.arange(*par_pose[k]) for k in garde])
        ok, rvec, tvec = cv2.solvePnP(obj[sel], img[sel], K, D, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        proj = cv2.projectPoints(obj, rvec, tvec, K, D)[0].reshape(-1, 2)
        err = np.linalg.norm(proj - img, axis=1)
        e_pose = {k: float(err[a:b].mean()) for k, (a, b) in par_pose.items()}
    R, _ = cv2.Rodrigues(rvec)
    T_ce = np.eye(4)
    T_ce[:3, :3], T_ce[:3, 3] = R, tvec.flatten()
    return np.linalg.inv(T_ce), e_pose, garde


def main():
    poses = lire_poses(os.path.join(HERE, "poses_d435.txt"))
    tags = json.load(open(os.path.join(HERE, "d435_tags.json")))
    T_bt = np.array(json.load(open(os.path.join(HERE, "T_base_tag0.json")))["T_base_tag0"])
    labels = [k for k in poses if k in tags]
    manquants = sorted(set(poses) ^ set(tags))
    print(f"{len(labels)} poses appariées : {labels}")
    if manquants:
        print(f"(étiquettes présentes d'un seul côté, ignorées : {manquants})")
    if len(labels) < 3:
        raise SystemExit("Il faut au moins 3 poses communes (idéalement 10-15).")

    X = {}
    for k in labels:
        T_be, _ = poses[k]
        T_ct = np.array(tags[k]["T_cam_tag0"])
        X[k] = np.linalg.inv(T_be) @ T_bt @ np.linalg.inv(T_ct)
    Xm = mean_pose(list(X.values()))

    print("\nÉcart de chaque pose à la moyenne :")
    dp, da = [], []
    for k in labels:
        p = 1000 * np.linalg.norm(X[k][:3, 3] - Xm[:3, 3])
        a = rot_angle_deg(Xm[:3, :3].T @ X[k][:3, :3])
        dp.append(p)
        da.append(a)
        print(f"  {k:6s} {p:6.1f} mm  {a:5.2f} deg   (tag vu à {np.linalg.norm(np.array(tags[k]['T_cam_tag0'])[:3, 3]):.2f} m)")
    print(f"  -> dispersion : {np.mean(dp):.1f} mm, {np.mean(da):.2f} deg en moyenne")

    print("\nMéthode robuste (solvePnP global sur les coins du tag de toutes les poses) :")
    res = pnp_global(labels, poses, tags, T_bt)
    methode = "moyenne des poses"
    if res is not None:
        X_pnp, e_pose, garde = res
        for k, e in e_pose.items():
            print(f"  {k:6s} erreur de reprojection {e:5.2f} px" + ("" if k in garde else "   (écartée)"))
        print(f"  -> erreur moyenne {np.mean([e_pose[k] for k in garde]):.2f} px sur {len(garde)} poses")
        print(f"  écart avec la méthode « moyenne des poses » : "
              f"{1000 * np.linalg.norm(X_pnp[:3, 3] - Xm[:3, 3]):.1f} mm, "
              f"{rot_angle_deg(Xm[:3, :3].T @ X_pnp[:3, :3]):.2f} deg")
        Xm = X_pnp
        methode = f"solvePnP global ({len(garde)} poses)"
    else:
        print("  impossible (images d435_calib/<etiquette>.png absentes ?) : on garde la moyenne des poses")

    # repère "hand" de MuJoCo = bride, même orientation que l'EE
    T_hand_ee = np.eye(4)
    T_hand_ee[2, 3] = EE_OFFSET
    T_hc = T_hand_ee @ Xm
    R = T_hc[:3, :3]
    W, H = tags[labels[0]]["res"]
    fy = np.array(tags[labels[0]]["K"])[1, 1]
    fovy = float(np.degrees(2 * np.arctan(H / 2 / fy)))
    out = {"T_EE_cam": Xm.round(6).tolist(), "T_hand_cam": T_hc.round(6).tolist(),
           "mujoco": {"pos": T_hc[:3, 3].round(5).tolist(),
                      "xyaxes": np.r_[R[:, 0], -R[:, 1]].round(5).tolist(),
                      "fovy": round(fovy, 3), "resolution": [W, H]},
           "methode": methode, "n_poses": len(labels), "dispersion_mm": round(float(np.mean(dp)), 2),
           "dispersion_deg": round(float(np.mean(da)), 3)}
    with open(os.path.join(HERE, "d435_pose_hand.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPose de la D435 dans le repère 'hand' (bride) : {np.round(T_hc[:3, 3] * 1000, 1)} mm, "
          f"fovy {fovy:.2f}° ({W}x{H})")
    print("-> perception/d435_pose_hand.json (relu par twin_scene.py)")

    superposer(labels, poses, tags)


def superposer(labels, poses, tags):
    """Rend la vue D435 simulée (pose calibrée, angles réels du robot) sur chaque image réelle."""
    import importlib

    import mujoco
    import twin_scene
    importlib.reload(twin_scene)                       # relit d435_pose_hand.json
    model = twin_scene.build_twin_model()
    data = mujoco.MjData(model)
    for k in labels:
        q = poses[k][1]
        img_path = os.path.join(HERE, "d435_calib", f"{k}.png")
        if q is None or not os.path.exists(img_path):
            continue
        real = cv2.imread(img_path)
        H, W = real.shape[:2]
        K = np.array(tags[k]["K"])
        D = np.array(tags[k]["D"])
        f = (H / 2) / np.tan(np.radians(twin_scene.D435_FOVY) / 2)
        Ki = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, Ki, (W, H), cv2.CV_32FC1)
        real_u = cv2.remap(real, m1, m2, cv2.INTER_LINEAR)
        twin_scene.set_arm(model, data, q)
        seg = twin_scene.render_webcam(model, data, W, H, segmentation=True, camera="d435").astype(np.int32)
        edges = np.zeros(seg.shape, np.uint8)
        edges[:, 1:][seg[:, 1:] != seg[:, :-1]] = 255
        edges[1:, :][seg[1:, :] != seg[:-1, :]] = 255
        over = real_u.copy()
        over[cv2.dilate(edges, np.ones((2, 2), np.uint8)) > 0] = (0, 255, 255)
        cv2.imwrite(os.path.join(HERE, "d435_calib", f"overlay_{k}.png"), over)
    print("Superpositions : perception/d435_calib/overlay_<etiquette>.png (jaune = sim)")


if __name__ == "__main__":
    main()
