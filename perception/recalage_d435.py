"""Recalage de la D435 au poignet sur la scène, à chaque photo (26/11).

La pose de la D435 vient de la chaîne  O_T_EE (robot, ~0,1 mm)  x  T_EE_cam (main-oeil du 23/11,
640x480, ~3-4 mm et quelques dixièmes de degré). À 85 cm de la table, 0,5° d'erreur sur T_EE_cam
déplace les points de ~7 mm : c'est le décalage de ~1 cm vu sur le cube le 26/11 (webcam / D435) et
la table vue 6 mm trop bas.

Depuis la pose perchée, la D435 voit la table rouge ET le tag 0, tous deux connus dans le repère
robot (plan mesuré au crayon, coins du tag au crayon à ~1 mm). On corrige donc la pose de la caméra
(6 paramètres, petite correction autour de la chaîne) pour que :
  - les 4 coins du tag tombent sur les coins détectés dans l'image couleur (erreur en pixels) ;
  - les points de la table (profondeur) tombent sur le plan de la table (erreur en mm).
La table fixe l'inclinaison et la hauteur (des milliers de points), le tag fixe x, y et le lacet.
Moindres carrés (Gauss-Newton). Garde-fous : tag vu, assez de points de table, correction < 3 cm / 3°,
résidus raisonnables ; sinon on garde la chaîne seule.

    recaler(prof_m, couleur_bgr, K, T_base_cam_chaine) -> (T_base_cam, info)
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from twin_scene import RED_PLANE, RED_X, RED_Y, T_BASE_TAG0, TAG_BLACK, red_z  # noqa: E402

CORR_MAX_M, CORR_MAX_DEG = 0.03, 3.0


def coins_tag_base():
    """4 coins du carré noir du tag 0 dans le repère robot, dans l'ordre des coins d'OpenCV."""
    h = TAG_BLACK / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
    return obj @ T_BASE_TAG0[:3, :3].T + T_BASE_TAG0[:3, 3]


def detecter_tag(couleur_bgr):
    """Coins (4, 2) du tag 0 dans l'image (sous-pixel), ou None."""
    g = cv2.cvtColor(couleur_bgr, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(g)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(g, d, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if i == 0:
            c = c.reshape(4, 1, 2).astype(np.float32)
            cv2.cornerSubPix(g, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
            return c.reshape(4, 2).astype(np.float64)
    return None


def _normale():
    a, b, _ = RED_PLANE
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


def _exp(x):
    """Petite correction (rotation vectorielle 3 + translation 3, repère caméra) -> matrice 4x4."""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], float))[0]
    T[:3, 3] = x[3:]
    return T


def points_table(prof_m, K, T_bc, n_max=3000, seuil=0.02, rng=None):
    """Points de profondeur (repère caméra) qui sont sur la table rouge, loin du tag et du trou."""
    rng = rng or np.random.default_rng(0)
    v, u = np.nonzero((prof_m > 0.25) & (prof_m < 2.0))
    if len(u) > 60000:
        i = rng.choice(len(u), 60000, replace=False)
        u, v = u[i], v[i]
    z = prof_m[v, u]
    Pc = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    Pb = Pc @ T_bc[:3, :3].T + T_bc[:3, 3]
    h = (Pb[:, 2] - red_z(Pb[:, 0], Pb[:, 1])) * _normale()[2]
    tag = T_BASE_TAG0[:2, 3]
    ok = ((Pb[:, 0] > RED_X[0] + 0.03) & (Pb[:, 0] < RED_X[1] - 0.03)
          & (Pb[:, 1] > RED_Y[0] + 0.03) & (Pb[:, 1] < RED_Y[1] - 0.03)
          & (np.abs(h) < seuil) & (np.hypot(Pb[:, 0] - tag[0], Pb[:, 1] - tag[1]) > 0.09)
          & (np.hypot(Pb[:, 0] - 0.319, Pb[:, 1] - 0.310) > 0.07))
    Pc = Pc[ok]
    if len(Pc) > n_max:
        Pc = Pc[rng.choice(len(Pc), n_max, replace=False)]
    return Pc


def _residus(x, T0, K, coins_px, coins_b, Pc, w_plan):
    T = T0 @ _exp(x)
    Tinv = np.linalg.inv(T)
    r = []
    if coins_px is not None:
        q = coins_b @ Tinv[:3, :3].T + Tinv[:3, 3]
        uv = np.c_[K[0, 0] * q[:, 0] / q[:, 2] + K[0, 2], K[1, 1] * q[:, 1] / q[:, 2] + K[1, 2]]
        r.append((uv - coins_px).ravel())                        # pixels
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    h = (Pb[:, 2] - red_z(Pb[:, 0], Pb[:, 1])) * _normale()[2]
    r.append(h / 0.003 * w_plan)                                 # ~3 mm de bruit
    return np.concatenate(r)


def recaler(prof_m, couleur_bgr, K, T_bc0):
    """Pose corrigée de la caméra (4x4, repère robot) et un dict d'info (ok, corrections, résidus)."""
    info = {"ok": False}
    coins_px = detecter_tag(couleur_bgr)
    info["tag_vu"] = coins_px is not None
    Pc = points_table(prof_m, K, T_bc0)
    info["points_table"] = int(len(Pc))
    if coins_px is None or len(Pc) < 500:
        info["raison"] = "tag 0 non vu" if coins_px is None else "pas assez de table visible"
        return T_bc0, info
    coins_b = coins_tag_base()
    w_plan = np.sqrt(30.0 / len(Pc))           # la table « pèse » comme ~30 mesures, le tag comme 8
    x = np.zeros(6)
    for it in range(2):
        for _ in range(15):
            r0 = _residus(x, T_bc0, K, coins_px, coins_b, Pc, w_plan)
            J = np.empty((len(r0), 6))
            for k in range(6):
                dx = np.zeros(6)
                dx[k] = 1e-6
                J[:, k] = (_residus(x + dx, T_bc0, K, coins_px, coins_b, Pc, w_plan) - r0) / 1e-6
            pas = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(6), -J.T @ r0, rcond=None)[0]
            x = x + pas
            if np.abs(pas).max() < 1e-7:
                break
        if it == 0:                            # 2e passe : points de table re-sélectionnés plus serrés
            Pc = points_table(prof_m, K, T_bc0 @ _exp(x), seuil=0.008)
            if len(Pc) < 500:
                break
            w_plan = np.sqrt(30.0 / len(Pc))
    T = T_bc0 @ _exp(x)
    r = _residus(x, T_bc0, K, coins_px, coins_b, Pc, w_plan)
    err_px = np.linalg.norm(r[:8].reshape(4, 2), axis=1)
    err_plan = np.abs(r[8:] / w_plan * 0.003)
    d_t = float(np.linalg.norm(T[:3, 3] - T_bc0[:3, 3]))
    d_r = float(np.degrees(np.arccos(np.clip((np.trace(T_bc0[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1))))
    # décalage induit au niveau de la table (ce qui compte pour les objets)
    centre = np.array([0.55, 0.0, 0.0])
    p0 = np.linalg.inv(T_bc0) @ np.r_[centre, 1]
    decal = float(np.linalg.norm((T @ p0)[:3] - centre) * 1000)
    info.update(correction_mm=round(d_t * 1000, 1), correction_deg=round(d_r, 2),
                decalage_table_mm=round(decal, 1), tag_px_moy=round(float(err_px.mean()), 2),
                plan_mm_med=round(float(np.median(err_plan) * 1000), 2), points_table=int(len(Pc)))
    if d_t > CORR_MAX_M or d_r > CORR_MAX_DEG or err_px.mean() > 4.0:
        info["raison"] = "correction trop grande ou tag mal ajusté : chaîne seule gardée"
        return T_bc0, info
    info["ok"] = True
    return T, info


def sauver(T_base_EE, T_bc, info):
    """Mémorise la main-oeil corrigée (T_EE_cam) : réutilisable aux autres poses du bras."""
    T_ec = np.linalg.inv(T_base_EE) @ T_bc
    with open(os.path.join(HERE, "d435_recalage.json"), "w") as f:
        json.dump({"T_EE_cam": np.round(T_ec, 6).tolist(), "info": info}, f, indent=1)


if __name__ == "__main__":
    # rejouer une capture : python perception/recalage_d435.py perception/d435_capture.npz
    z = np.load(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "d435_capture.npz"))
    with open(os.path.join(HERE, "d435_pose_hand.json")) as f:
        T_ec = np.array(json.load(f)["T_EE_cam"])
    T0 = z["T_base_EE"] @ T_ec
    T, info = recaler(z["prof_m"], z["couleur"], z["K"], T0)
    print(info)
