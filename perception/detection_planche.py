"""Détection de la scène avec obstacle : planche (posture, x, y, lacet), cube et cible -> repère robot.

Deux caméras, deux méthodes indépendantes (on compare les deux, puis on vérifie au crayon) :

  WEBCAM (globale, fixe) — « analyse par synthèse », comme ajuster_cube de detection.py :
    1. image remappée sur la caméra idéale ;
    2. masque de la planche par la TEINTE (aggloméré orangé : écart de teinte au rouge de la table
       entre 6 et 22, saturation > 60), plus grosse tache dont le pied tombe sur la table ;
    3. on cherche la boîte 53,58 x 10 x 3 cm (posture, x, y, lacet) dont la silhouette projetée
       recouvre le mieux la tache (IoU). La partie de la planche qui sort de l'image (haut du
       poteau debout) est hors image pour le modèle aussi : elle ne compte pas.
    Cube et cible : detection.py (inchangé), sur la même image.

  D435 (au poignet) — géométrie pure, par la PROFONDEUR :
    1. profondeur alignée sur la couleur (médiane de 15 images) -> nuage de points caméra ;
    2. pose de la caméra = O_T_EE (lu sur le robot par lien_robot.etat) . T_EE_cam (calibration
       main-oeil d435_pose_hand.json) -> nuage dans le repère robot ;
    3. hauteur de chaque point au-dessus du plan de la table rouge (red_z) ; ce qui dépasse de
       1,2 cm = objets ; regroupement sur une grille de 5 mm au sol ;
    4. planche = l'amas orangé (ou le plus grand) : sa HAUTEUR donne la posture (3 / 10 / 54 cm),
       le rectangle minimal de sa face du DESSUS donne x, y et le lacet ;
       cube = l'amas bleu (~4 cm), cible = pixels verts au niveau de la table.

Usage (PowerShell, venv activé, à la racine du dépôt) :
    python perception/detection_planche.py                      webcam en direct (ESPACE = enregistrer)
    python perception/detection_planche.py --image perception/vue_webcam.png
    python perception/detection_planche.py --d435 --ip 10.254.79.65
        D435 seule : amener d'abord le bras (à la main) en pose d'observation, pince vers le bas,
        assez haut (caméra à > 35 cm du haut de la planche : ~95 cm au-dessus de la table si elle
        est debout), toute la scène dans le champ ;
    python perception/detection_planche.py --les-deux --ip 10.254.79.65
        webcam + D435 : à CHAQUE ESPACE, nouvelle photo D435 + détection webcam, écart entre les deux ; la scène retenue prend la posture de la
        D435 (mesure directe de la hauteur) et la moyenne des positions.
    Options : --posture plat|tranche|debout (imposer la posture à la webcam),
              --hors-ligne perception/d435_capture.npz (rejouer une capture D435 enregistrée, sans
              robot ni caméra ; combinable avec --image pour refaire la fusion).
Sorties : perception/detection_planche.json (clé "scene" = dictionnaire au format du jumeau :
          cube_xy, cube_lacet_deg, cible_xy, posture, planche_xy, planche_lacet),
          perception/detection_planche.png (webcam annotée), perception/detection_planche_d435.png,
          perception/d435_capture.npz (dernière capture D435 : profondeur, couleur, pose).
"""
import argparse
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RACINE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RACINE, "common"))
sys.path.insert(0, os.path.join(RACINE, "avec_obstacle", "common"))     # planche.py
sys.path.insert(0, os.path.join(RACINE, "jumeau"))
from planche import (EMPREINTE, HAUTEUR, PLANCHE_EP, PLANCHE_L, PLANCHE_LARG,  # noqa: E402
                     POSTURES, pose_planche)
from twin_scene import CUBE_HALF, RED_PLANE, RED_X, RED_Y, red_z  # noqa: E402

W, H = 1280, 720
DEMI = np.array([PLANCHE_L / 2, PLANCHE_LARG / 2, PLANCHE_EP / 2])
SORTIE_JSON = os.path.join(HERE, "detection_planche.json")

# ---------------------------------------------------------------- segmentation
# Mesuré au labo (26/11). La couleur de la webcam DÉRIVE (balance des blancs auto, soleil) : la
# teinte médiane de la table rouge est passée de -1 à +5 dans la même après-midi, et la planche de
# 7-11 à 13-14. Ce qui reste stable, c'est l'ÉCART : la planche est 5 à 15 crans plus orange que la
# table, et nettement moins saturée (S 37-120 contre ~135-160 pour la table). Les deux seuils sont
# donc pris RELATIVEMENT à la table, mesurée sur chaque image (médiane dans le polygone de la table).
# Recherche limitée à la table rouge (le mur et le sol beiges du fond ont la teinte de la planche).
ECART_TEINTE = (6, 22)              # teinte planche - teinte table (OpenCV, 0-180)
SAT_MIN = 25
SAT_MAX_REL = 1.5                   # garde-fou seulement (la saturation de la planche varie de 37 à 175)
AIRE_MIN_PLANCHE = 1500             # pixels


def masque_table(cam, marge=0.005):
    """Polygone de la table rouge projeté dans l'image webcam corrigée (255 dedans)."""
    m = np.zeros((H, W), np.uint8)
    coins = [(RED_X[0] + marge, RED_Y[0] + marge), (RED_X[1] - marge, RED_Y[0] + marge),
             (RED_X[1] - marge, RED_Y[1] - marge), (RED_X[0] + marge, RED_Y[1] - marge)]
    pts = []
    for (x0, y0), (x1, y1) in zip(coins, coins[1:] + coins[:1]):      # bords échantillonnés
        for t in np.linspace(0, 1, 20, endpoint=False):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            pts.append(cam.point_vers_pixel(np.array([x, y, red_z(x, y)])))
    cv2.fillPoly(m, [np.int32(pts)], 255)
    return m


def masque_planche(img_c, table=None):
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_c, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    h = np.where(h > 90, h - 180, h)                 # teinte signée autour du rouge (-90..90)
    sat = hsv[:, :, 1]
    zone = table > 0 if table is not None else np.ones(h.shape, bool)
    rouge = zone & (np.abs(h) < 20) & (sat > 90)     # pixels « table » pour la référence
    h_t = float(np.median(h[rouge])) if rouge.sum() > 1000 else 0.0
    s_t = float(np.median(sat[rouge])) if rouge.sum() > 1000 else 150.0
    d = h - h_t
    m = ((d >= ECART_TEINTE[0]) & (d <= ECART_TEINTE[1]) & (sat > SAT_MIN)
         & (sat < SAT_MAX_REL * s_t)).astype(np.uint8) * 255
    if table is not None:
        m &= table
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return m


# ---------------------------------------------------------------- modèle 3D de la planche
def _quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_SIGNES = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def coins_planche(posture, x, y, lacet):
    """Les 8 coins 3D (repère robot) de la planche, même convention que le jumeau (pose_planche)."""
    pq = pose_planche(posture, x, y, lacet)
    return pq[:3] + (_SIGNES * DEMI) @ _quat_mat(pq[3:]).T


def normale_table():
    a, b, _ = RED_PLANE
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


# ---------------------------------------------------------------- WEBCAM : ajustement de la boîte
class AjusteurWebcam:
    """IoU entre la tache de la planche et la silhouette projetée de la boîte, sur une image réduite
    (facteur ech). Calculée seulement DANS LA ZONE UTILE (image de la table rouge) : ce qui sort du
    cadre ou de la table (haut d'un poteau debout devant le mur) est ignoré pour le modèle comme
    pour la vraie image."""

    def __init__(self, cam, masque, ech, zone=None):
        self.cam, self.ech = cam, ech
        self.cible = (cv2.resize(masque, None, fx=ech, fy=ech, interpolation=cv2.INTER_AREA) > 127)
        self.cible = self.cible.astype(np.uint8)
        if zone is None:
            zone = np.full(masque.shape, 255, np.uint8)
        self.zone = (cv2.resize(zone, None, fx=ech, fy=ech, interpolation=cv2.INTER_NEAREST) > 127)
        self.zone = self.zone.astype(np.uint8)
        self.cible &= self.zone
        self.n_c = int(self.cible.sum())
        self.buf = np.zeros_like(self.cible)
        R = np.linalg.inv(cam.T_bc)
        self.Rcb, self.tcb = R[:3, :3], R[:3, 3]

    def silhouette(self, posture, x, y, lacet):
        P = coins_planche(posture, x, y, lacet) @ self.Rcb.T + self.tcb
        if np.any(P[:, 2] < 0.05):
            return None
        K = self.cam.Ki
        uv = np.c_[K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2], K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]]
        return cv2.convexHull((uv * self.ech).astype(np.float32)).astype(np.int32)

    def score(self, posture, x, y, lacet):
        sil = self.silhouette(posture, x, y, lacet)
        if sil is None:
            return 0.0
        self.buf[:] = 0
        cv2.fillConvexPoly(self.buf, sil, 1)
        self.buf &= self.zone
        inter = int(np.count_nonzero(self.buf & self.cible))
        union = self.n_c + int(np.count_nonzero(self.buf)) - inter
        return inter / union if union else 0.0


def _grille(aj, posture, x0, y0, l0, rayon, pas, dl, pas_l, best):
    for lac in l0 + np.radians(np.arange(-dl, dl + 1e-9, pas_l)):
        for dx in np.arange(-rayon, rayon + 1e-9, pas):
            for dy in np.arange(-rayon, rayon + 1e-9, pas):
                sc = aj.score(posture, x0 + dx, y0 + dy, lac)
                if sc > best[0]:
                    best = (sc, x0 + dx, y0 + dy, lac)
    return best


def ajuster_planche(cam, masque, contour, postures=POSTURES, zone=None, depart=None):
    """Renvoie {'posture', 'x', 'y', 'lacet' (rad, 0-pi), 'iou', 'iou_par_posture'}.
    depart : estimation précédente (suivi en continu) -> recherche locale, ~20x plus rapide."""
    grossier = AjusteurWebcam(cam, masque, 0.25, zone)
    fin = AjusteurWebcam(cam, masque, 0.5, zone)
    resultats = {}
    if depart is not None:
        po = depart["posture"]
        best = (-1.0, depart["x"], depart["y"], depart["lacet"])
        best = _grille(grossier, po, best[1], best[2], best[3], 0.02, 0.005, 6, 2, best)
        best = (fin.score(po, *best[1:]), *best[1:])
        best = _grille(fin, po, best[1], best[2], best[3], 0.004, 0.001, 1.5, 0.5, best)
        resultats[po] = best
    else:
        mo = cv2.moments(contour)
        u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        pied = contour.reshape(-1, 2)[np.argmax(contour.reshape(-1, 2)[:, 1])]     # point le plus bas
        for po in postures:
            if po == "debout":
                # le haut du poteau peut sortir de l'image : on part du pied de la tache, au sol
                p0 = cam.pixel_vers_plan(float(pied[0]), float(pied[1]) - 15, 0.0)
            else:
                p0 = cam.pixel_vers_plan(u, v, HAUTEUR[po] / 2)
            best = (-1.0, p0[0], p0[1], 0.0)
            best = _grille(grossier, po, p0[0], p0[1], np.pi / 2, 0.12, 0.02, 90, 6, best)
            best = _grille(grossier, po, best[1], best[2], best[3], 0.025, 0.005, 6, 2, best)
            best = (fin.score(po, *best[1:]), *best[1:])
            best = _grille(fin, po, best[1], best[2], best[3], 0.006, 0.002, 2, 0.5, best)
            best = _grille(fin, po, best[1], best[2], best[3], 0.0015, 0.0005, 0.5, 0.25, best)
            resultats[po] = best
    po = max(resultats, key=lambda k: resultats[k][0])
    sc, x, y, lac = resultats[po]
    return {"posture": po, "x": float(x), "y": float(y), "lacet": float(lac % np.pi),
            "iou": round(float(sc), 3),
            "iou_par_posture": {k: round(float(r[0]), 3) for k, r in resultats.items()}}


def planche_webcam(img_c, cam, table=None, postures=POSTURES, depart=None):
    """Planche seule sur une image webcam corrigée : (résultat ou None, contour). depart : voir
    ajuster_planche ; si l'ajustement local est mauvais (IoU < 0,8), on refait la recherche complète."""
    table = masque_table(cam) if table is None else table
    masque = masque_planche(img_c, table)
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for ct in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(ct) < AIRE_MIN_PLANCHE:
            break
        pts = ct.reshape(-1, 2)
        pied = pts[np.argmax(pts[:, 1])]
        p = cam.pixel_vers_plan(float(pied[0]), float(pied[1]), 0.0)
        if RED_X[0] - 0.05 < p[0] < RED_X[1] + 0.05 and RED_Y[0] - 0.05 < p[1] < RED_Y[1] + 0.05:
            # on ne garde que la tache retenue dans le masque (pas le reste orangé de l'image)
            seul = np.zeros_like(masque)
            cv2.drawContours(seul, [ct], -1, 255, -1)
            pl = ajuster_planche(cam, seul, ct, postures, table, depart=depart)
            if depart is not None and pl["iou"] < 0.8:
                pl = ajuster_planche(cam, seul, ct, postures, table)
            pl["aire_px"] = int(cv2.contourArea(ct))
            pl["bord_image"] = bool(pts[:, 1].min() <= 2 or pts[:, 0].min() <= 2 or pts[:, 0].max() >= W - 3)
            return pl, ct
    return None, None


def detecter_webcam(img_c, cam, postures=POSTURES):
    """Planche (ajustement) + cube et cible (detection.py) sur une image webcam corrigée."""
    import detection as det_cube
    planche, contour_pl = planche_webcam(img_c, cam, None, postures)
    res_cube, _ = det_cube.detecter(img_c, cam, det_cube.charger_seuils())
    return {"planche": planche, "cube": res_cube.get("cube"), "cible": res_cube.get("cible")}, contour_pl


def annoter_webcam(img_c, res, contour_pl, cam):
    import detection as det_cube
    out = det_cube.annoter(img_c, {k: res[k] for k in ("cube", "cible")}, cam)
    pl = res["planche"]
    if pl is None:
        cv2.putText(out, "planche : non trouvee", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    cv2.drawContours(out, [contour_pl], -1, (0, 200, 255), 2)
    C = coins_planche(pl["posture"], pl["x"], pl["y"], pl["lacet"])
    uv = np.array([cam.point_vers_pixel(c) for c in C])
    for i in range(8):                    # arêtes de la boîte ajustée (coins voisins : 1 signe change)
        for j in range(i + 1, 8):
            if np.sum(_SIGNES[i] != _SIGNES[j]) == 1:
                cv2.line(out, tuple(np.int32(uv[i])), tuple(np.int32(uv[j])), (255, 255, 255), 1)
    txt = (f"planche {pl['posture']} : x={pl['x']:.3f} y={pl['y']:.3f} m  lacet {np.degrees(pl['lacet']):.0f} deg"
           f"  IoU {pl['iou']:.2f}")
    cv2.putText(out, txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    return out


# ---------------------------------------------------------------- D435 : nuage de points
def nuage_base(prof_m, K, T_base_cam):
    """Profondeur (m, alignée sur la couleur) -> points (N, 3) dans le repère robot + indices pixels."""
    v, u = np.nonzero((prof_m > 0.20) & (prof_m < 2.0))
    z = prof_m[v, u]
    P = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    return P @ T_base_cam[:3, :3].T + T_base_cam[:3, 3], u, v


def _teinte_amas(hsv_px):
    """Classe de couleur d'un amas : 'planche' (orangé), 'cube' (bleu) ou None."""
    h, s = hsv_px[:, 0].astype(np.int16), hsv_px[:, 1]
    sat = (s > 40) & (hsv_px[:, 2] > 100)          # les zones sombres paraissent bleues sur la D435
    if sat.sum() < 20:
        return None, 0.0
    f_pl = float(np.mean((h[sat] >= 5) & (h[sat] <= 22)))
    f_cu = float(np.mean((h[sat] >= 95) & (h[sat] <= 130)))
    if max(f_pl, f_cu) < 0.4:
        return None, max(f_pl, f_cu)
    return ("planche", f_pl) if f_pl > f_cu else ("cube", f_cu)


def posture_depuis_hauteur(h):
    if h < 0.065:
        return "plat"
    if h < 0.30:
        return "tranche"
    return "debout"


def rect_robuste(pts, periode=180.0, pas=0.5):
    """Rectangle d'aire minimale qui contient 99 % des points (0,5e-99,5e centiles sur chaque axe) :
    moins sensible que cv2.minAreaRect aux points isolés (bruit de profondeur, bords).
    Renvoie centre (2), (grand côté, petit côté), angle du grand côté (deg, 0-180)."""
    pts = np.asarray(pts, float)
    best = None
    for a in np.arange(0, periode, pas):
        c, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
        u, v = pts @ [c, s_], pts @ [-s_, c]
        u0, u1 = np.percentile(u, (0.5, 99.5))
        v0, v1 = np.percentile(v, (0.5, 99.5))
        aire = (u1 - u0) * (v1 - v0)
        if best is None or aire < best[0]:
            best = (aire, a, (u0 + u1) / 2, (v0 + v1) / 2, u1 - u0, v1 - v0)
    _, a, um, vm, lu, lv = best
    c, s_ = np.cos(np.radians(a)), np.sin(np.radians(a))
    centre = um * np.array([c, s_]) + vm * np.array([-s_, c])
    if lu < lv:
        return centre, (float(lv), float(lu)), float((a + 90) % 180)
    return centre, (float(lu), float(lv)), float(a)


def completer_faces_cachees(centre, dims, ang_deg, dims_att, cam_xy):
    """On ne voit que les faces tournées vers la caméra : si l'emprise mesurée est plus courte que
    la vraie, la partie manquante est du côté opposé à la caméra -> on y décale le centre."""
    a = np.radians(ang_deg)
    axes = (np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)]))
    c = np.array(centre, float)
    for ax, mes, att in zip(axes, dims, dims_att):
        manque = att - mes
        if manque > 0.002:
            sens = -np.sign((cam_xy - c) @ ax) or 1.0
            c = c + sens * ax * manque / 2
    return c


CELL = 0.005           # grille au sol (m)
H_OBJET = 0.012        # au-dessus de la table de plus de 12 mm = objet


def detecter_d435(prof_m, couleur_bgr, K, T_base_cam, ref=None):
    """Planche, cube et cible à partir d'UNE vue D435 (profondeur alignée + couleur).
    ref : planche vue par la webcam ({"x", "y", ...}) -> on prend l'amas D435 le plus proche."""
    P, u, v = nuage_base(prof_m, K, T_base_cam)
    hsv = cv2.cvtColor(couleur_bgr, cv2.COLOR_BGR2HSV)[v, u]
    dz = P[:, 2] - red_z(P[:, 0], P[:, 1])
    n = normale_table()
    hauteur = dz * n[2]                               # distance au plan incliné
    sur_table = ((P[:, 0] > RED_X[0] + 0.01) & (P[:, 0] < RED_X[1] - 0.01)
                 & (P[:, 1] > RED_Y[0] + 0.01) & (P[:, 1] < RED_Y[1] - 0.01))
    plan = sur_table & (np.abs(hauteur) < 0.006)
    info = {"points": int(len(P)), "points_table": int(plan.sum())}
    if plan.sum() > 500:                              # contrôle de la calibration : la table est-elle à 0 ?
        info["table_ecart_mm"] = round(float(np.median(hauteur[sur_table & (np.abs(hauteur) < 0.02)])) * 1000, 1)
    obj = sur_table & (hauteur > H_OBJET) & (hauteur < 0.70)
    res = {"planche": None, "cube": None, "cible": None, "info": info}

    # cible : pixels verts au niveau de la table
    hh = hsv[:, 0].astype(np.int16)
    # (couleurs de la D435 très bleutées au labo : blanc du tag H ~89 S ~120, cible H ~91 S 255)
    # -> on prend la plus grosse tache verte-cyan TRÈS saturée, de la taille d'un disque de 6 cm
    vert_px = np.zeros(prof_m.shape, np.uint8)
    sel_v = plan & (hh >= 55) & (hh <= 97) & (hsv[:, 1] > 170) & (hsv[:, 2] > 140)   # V : pas le noir du tag
    vert_px[v[sel_v], u[sel_v]] = 255
    vert_px = cv2.morphologyEx(vert_px, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_v, lab_v, st_v, _ = cv2.connectedComponentsWithStats(vert_px)
    if n_v > 1:
        k = 1 + int(np.argmax(st_v[1:, 4]))
        sel = sel_v & (lab_v[v, u] == k)
        if sel.sum() > 50:
            c = np.median(P[sel, :2], axis=0)
            res["cible"] = {"xyz": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(red_z(*c)), 4)],
                            "points": int(sel.sum())}
    if obj.sum() < 30:
        return res

    Po, ho, hsvo = P[obj], hauteur[obj], hsv[obj]
    x0, y0 = RED_X[0], RED_Y[0]
    ix = ((Po[:, 0] - x0) / CELL).astype(int)
    iy = ((Po[:, 1] - y0) / CELL).astype(int)
    nx, ny = int((RED_X[1] - x0) / CELL) + 1, int((RED_Y[1] - y0) / CELL) + 1
    occ = np.zeros((ny, nx), np.uint8)
    occ[iy, ix] = 255
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n_lab, lab = cv2.connectedComponents(occ)
    lab_pt = lab[iy, ix]
    amas = []
    for k in range(1, n_lab):
        sel = lab_pt == k
        if sel.sum() < 40:
            continue
        classe, frac = _teinte_amas(hsvo[sel])
        hmax = float(np.percentile(ho[sel], 97))
        amas.append({"sel": sel, "classe": classe, "frac": frac, "hmax": hmax,
                     "aire_cm2": float(np.count_nonzero(lab == k)) * CELL ** 2 * 1e4})
    if not amas:
        return res
    info["amas"] = [{"classe": a["classe"], "hauteur_cm": round(a["hmax"] * 100, 1),
                     "aire_cm2": round(a["aire_cm2"], 1)} for a in amas]

    # --- planche. Plusieurs amas peuvent ressembler à une planche (le boîtier de commande Franka posé
    # sur la table, 8 cm de haut, vu le 26/11 depuis une pose trop à droite). On analyse chaque amas
    # comme une planche, puis on choisit : près de l'estimation WEBCAM si on l'a (ref), sinon celui
    # dont la forme colle le mieux aux dimensions de la vraie planche.
    def est_cube(a):
        return a["classe"] == "cube" or (a["classe"] is None and 0.03 < a["hmax"] < 0.055
                                          and a["aire_cm2"] < 40)

    def analyser(a):
        Pa, ha = Po[a["sel"]], ho[a["sel"]]
        # chaque point ramené à la verticale de son pied (la planche est perpendiculaire à la
        # table inclinée) : p_sol = p_xy - n_xy.h -> tous les points vus tombent dans l'empreinte
        sol = (Pa[:, :2] - np.outer(ha, n[:2])).astype(np.float32)
        dessus = ha > a["hmax"] - 0.012
        (cx, cy), (lx, ly), ang = rect_robuste(sol[dessus])
        po = posture_depuis_hauteur(a["hmax"])
        tronque = False
        if po != "debout" and lx < 0.25 and a["hmax"] > 0.15:
            # haut d'un poteau trop près de la caméra (sous la portée minimale) : on ne voit que
            # les flancs, sa « hauteur » est fausse mais son empreinte est petite -> debout
            po, tronque = "debout", True
        if po == "debout":
            # flancs + dessus éventuel ; seules les faces tournées vers la caméra sont vues
            (cx, cy), (lx, ly), ang = rect_robuste(sol[ha > 0.03])
            cx, cy = completer_faces_cachees((cx, cy), (lx, ly), ang, EMPREINTE["debout"], T_base_cam[:2, 3])
        lacet = np.radians(ang) % np.pi
        L_att, l_att = EMPREINTE[po]
        a_res = {"posture": po, "x": float(cx), "y": float(cy), "lacet": float(lacet),
                 "hauteur_mesuree_cm": round(a["hmax"] * 100, 1),
                 "dessus_mesure_cm": [round(lx * 100, 1), round(ly * 100, 1)],
                 "dessus_attendu_cm": [round(L_att * 100, 1), round(l_att * 100, 1)],
                 "couleur": a["classe"]}
        if tronque:
            a_res["attention"] = "haut du poteau trop près de la D435 (non mesuré) : monter le bras"
        elif po != "debout" and lx < L_att - 0.03:
            # planche coupée par le bord de l'image : le bout VISIBLE est fiable (le centre ne l'est
            # pas) -> on garde la direction et la position latérale, et on place le centre à L/2
            # du bout visible, vers le bout coupé
            ax = np.array([np.cos(lacet), np.sin(lacet)])
            bouts = [np.array([cx, cy]) + sg * lx / 2 * ax for sg in (1, -1)]
            Rcb = np.linalg.inv(T_base_cam)
            Hh, Ww = prof_m.shape

            def marge_bord(b):
                Hp = HAUTEUR[po]
                pb = np.r_[b + n[:2] * Hp, red_z(*b) + n[2] * Hp]
                q = Rcb[:3, :3] @ pb + Rcb[:3, 3]
                uu, vv = K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]
                return min(uu, vv, Ww - 1 - uu, Hh - 1 - vv)

            mb = [marge_bord(b) for b in bouts]
            if min(mb) < 25 <= max(mb):
                i_vu = int(np.argmax(mb))
                vers_coupe = bouts[1 - i_vu] - bouts[i_vu]
                vers_coupe /= np.linalg.norm(vers_coupe)
                c_new = bouts[i_vu] + vers_coupe * L_att / 2
                a_res["x"], a_res["y"] = float(c_new[0]), float(c_new[1])
                a_res["attention"] = ("planche coupée par le bord de l'image : centre recalé depuis le bout "
                                      "visible (moins sûr que la webcam)")
            else:
                a_res["attention"] = "empreinte plus courte que prévu : planche en partie hors champ ou cachée"
        return a_res

    def forme_plausible(r):
        lx, ly = r["dessus_mesure_cm"][0] / 100, r["dessus_mesure_cm"][1] / 100
        po, coupe = r["posture"], "attention" in r
        if po == "debout":
            return lx < 0.16 and ly < 0.08
        if po == "tranche":
            return ly < 0.06 and (lx > 0.25 or coupe)
        return 0.06 < ly < 0.14 and (lx > 0.25 or coupe)

    cand = [a for a in amas if not est_cube(a) and a["aire_cm2"] > 25]
    analyses = []
    for a in cand:
        try:
            analyses.append((a, analyser(a)))
        except Exception:   # noqa: BLE001
            continue
    choix = None
    if ref is not None:
        proches = [(np.hypot(r["x"] - ref["x"], r["y"] - ref["y"]), r) for _, r in analyses]
        proches = [x for x in proches if x[0] < 0.15]
        if proches:
            choix = min(proches, key=lambda x: x[0])[1]
            choix["choisi"] = "près de l'estimation webcam"
        else:
            info["planche_rejet"] = "aucun amas D435 à moins de 15 cm de la planche vue par la webcam"
    else:
        ok = [(a, r) for a, r in analyses if forme_plausible(r)]
        if ok:
            choix = max(ok, key=lambda x: x[0]["aire_cm2"])[1]
            choix["choisi"] = "forme compatible avec la planche"
        elif analyses:
            info["planche_rejet"] = "aucun amas D435 n'a la forme de la planche"
    if choix is not None:
        res["planche"] = choix

    # --- cube : amas bleu de ~4 cm
    cubes = [a for a in amas if est_cube(a) and a["hmax"] < 0.07]
    if cubes:
        a = max(cubes, key=lambda a: (a["classe"] == "cube", a["frac"]))
        Pa, ha = Po[a["sel"]], ho[a["sel"]]
        dessus = ha > a["hmax"] - 0.008
        xy, _, ang = rect_robuste(Pa[dessus, :2] - n[:2] * 2 * CUBE_HALF, periode=90.0)
        res["cube"] = {"xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4),
                               round(float(red_z(*xy) + CUBE_HALF), 4)],
                       "lacet_deg": round(float(ang % 90), 1), "hauteur_mesuree_cm": round(a["hmax"] * 100, 1)}
    return res


def T_EE_cam():
    with open(os.path.join(HERE, "d435_pose_hand.json")) as f:
        return np.array(json.load(f)["T_EE_cam"])


def capturer_d435(ip, n_img=15):
    """Une vue D435 (médiane de n_img profondeurs alignées) + la pose du robot au même moment."""
    import pyrealsense2 as rs
    from lien_robot import etat
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    echelle = prof.get_device().first_depth_sensor().get_depth_scale()
    for capteur in prof.get_device().query_sensors():          # balance des blancs automatique
        if capteur.supports(rs.option.enable_auto_white_balance):
            capteur.set_option(rs.option.enable_auto_white_balance, 1)
    align = rs.align(rs.stream.color)
    for _ in range(60):                                   # exposition auto
        pipe.wait_for_frames()
    e1 = etat(ip)
    pile, couleur = [], None
    for _ in range(n_img):
        fr = align.process(pipe.wait_for_frames())
        pile.append(np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32))
        couleur = np.asanyarray(fr.get_color_frame().get_data()).copy()
        intr = fr.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
    e2 = etat(ip)
    pipe.stop()
    p1, p2 = np.array(e1["O_T_EE"])[12:15], np.array(e2["O_T_EE"])[12:15]
    if np.linalg.norm(p1 - p2) > 0.002:
        raise SystemExit("[ERREUR] le bras a bougé pendant la capture : le laisser immobile et relancer")
    pile = np.array(pile)
    pile[pile == 0] = np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof_m = np.nan_to_num(np.nanmedian(pile, axis=0)) * echelle
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    T_base_EE = np.array(e2["O_T_EE"]).reshape(4, 4).T          # libfranka : colonne-major
    np.savez_compressed(os.path.join(HERE, "d435_capture.npz"), prof_m=prof_m, couleur=couleur, K=K,
                        T_base_EE=T_base_EE, q=np.array(e2["q"]))
    print("[d435] capture -> perception/d435_capture.npz")
    return prof_m, couleur, K, T_base_EE


def annoter_d435(couleur, res, K, T_base_cam):
    out = couleur.copy()
    Rcb = np.linalg.inv(T_base_cam)

    def px(p):
        q = Rcb[:3, :3] @ p + Rcb[:3, 3]
        return (int(K[0, 0] * q[0] / q[2] + K[0, 2]), int(K[1, 1] * q[1] / q[2] + K[1, 2])) if q[2] > 0.05 else None

    pl = res["planche"]
    if pl:
        C = coins_planche(pl["posture"], pl["x"], pl["y"], pl["lacet"])
        uv = [px(c) for c in C]
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(_SIGNES[i] != _SIGNES[j]) == 1 and uv[i] and uv[j]:
                    cv2.line(out, uv[i], uv[j], (0, 200, 255), 2)
        cv2.putText(out, f"planche {pl['posture']} x={pl['x']:.3f} y={pl['y']:.3f} lacet "
                    f"{np.degrees(pl['lacet']):.0f} h={pl['hauteur_mesuree_cm']} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    for nom, col, y_txt in (("cube", (255, 120, 0), 60), ("cible", (0, 200, 0), 90)):
        r = res.get(nom)
        if r:
            p = px(np.array(r["xyz"]))
            if p:
                cv2.drawMarker(out, p, col, cv2.MARKER_CROSS, 24, 2)
            cv2.putText(out, f"{nom} x={r['xyz'][0]:.3f} y={r['xyz'][1]:.3f}", (10, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


# ---------------------------------------------------------------- fusion + sortie
def ecart_lacet_deg(a, b, periode=np.pi):
    d = (a - b) % periode
    return float(np.degrees(min(d, periode - d)))


def scene_depuis(webcam, d435):
    """Dictionnaire de scène au format du jumeau (TwinPlancheEnv.reset(options={'scene': ...}))."""
    sc, ecarts = {}, {}
    pw = webcam["planche"] if webcam else None
    pd = d435["planche"] if d435 else None
    if pw and pd:
        ecarts["planche_xy_mm"] = round(float(np.hypot(pw["x"] - pd["x"], pw["y"] - pd["y"])) * 1000, 1)
        ecarts["planche_lacet_deg"] = round(ecart_lacet_deg(pw["lacet"], pd["lacet"]), 1)
        ecarts["posture"] = "identique" if pw["posture"] == pd["posture"] else f"{pw['posture']} / {pd['posture']}"
        po = pd["posture"]                              # hauteur mesurée : plus sûr
        fiable_d435 = not pd.get("attention") and (po != "debout" or pd["hauteur_mesuree_cm"] > 45)
        if fiable_d435:
            # COUPLAGE (26/11) : la webcam trouve la planche et sert de garde-fou (l'amas D435 retenu
            # est celui qui est près d'elle) ; la D435, qui MESURE la 3D (recalée sur le tag et la
            # table), donne la posture, le lacet et pèse 3/4 dans la position
            w = 0.75
            p = {"posture": po, "x": w * pd["x"] + (1 - w) * pw["x"], "y": w * pd["y"] + (1 - w) * pw["y"],
                 "lacet": pd["lacet"]}
            ecarts["position_retenue"] = "D435 3/4 + webcam 1/4, lacet D435"
        else:
            # planche en partie hors champ de la D435, ou poteau dont elle ne voit pas le haut :
            # la silhouette webcam est plus sûre pour la position
            p = {"posture": po, "x": pw["x"], "y": pw["y"], "lacet": pw["lacet"]}
            ecarts["position_retenue"] = "webcam (D435 incomplète)"
    else:
        p = pw or pd
    if p:
        sc.update(posture=p["posture"], planche_xy=[round(p["x"], 4), round(p["y"], 4)],
                  planche_lacet=round(p["lacet"], 4))
    for nom in ("cube", "cible"):
        rw = webcam.get(nom) if webcam else None
        rd = d435.get(nom) if d435 else None
        r = rw or rd                       # webcam d'abord : corrigée au robot (correction_detection.json)
        if rw and rd:
            ecarts[f"{nom}_xy_mm"] = round(float(np.hypot(rw["xyz"][0] - rd["xyz"][0],
                                                          rw["xyz"][1] - rd["xyz"][1])) * 1000, 1)
        if r:
            sc[f"{nom}_xy"] = [round(r["xyz"][0], 4), round(r["xyz"][1], 4)]
            if nom == "cube":
                sc["cube_lacet_deg"] = r.get("lacet_deg", 0.0)
    return sc, ecarts


def propre(d):
    if d is None:
        return None
    return {k: (propre(v) if isinstance(v, dict) else v) for k, v in d.items()
            if k not in ("contour", "sel")}


def enregistrer(webcam, d435, img_w=None, img_d=None):
    sc, ecarts = scene_depuis(webcam, d435)
    out = {"scene": sc, "ecarts_webcam_d435": ecarts, "webcam": propre(webcam), "d435": propre(d435),
           "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(SORTIE_JSON, "w") as f:
        json.dump(out, f, indent=2, default=float)
    if img_w is not None:
        cv2.imwrite(os.path.join(HERE, "detection_planche.png"), img_w)
    if img_d is not None:
        cv2.imwrite(os.path.join(HERE, "detection_planche_d435.png"), img_d)
    for src, r in (("webcam", webcam), ("d435", d435)):
        if r is None:
            continue
        pl = r["planche"]
        if pl:
            extra = (f"IoU {pl['iou']} {pl['iou_par_posture']}" if "iou" in pl
                     else f"hauteur {pl['hauteur_mesuree_cm']} cm, dessus {pl['dessus_mesure_cm']} cm")
            print(f"  [{src:6s}] planche {pl['posture']:7s} x = {pl['x']:.4f}  y = {pl['y']:.4f} m  "
                  f"lacet {np.degrees(pl['lacet']):6.1f} deg   ({extra})")
            if pl.get("attention"):
                print(f"           ATTENTION : {pl['attention']}")
        else:
            print(f"  [{src:6s}] planche non trouvée")
        for nom in ("cube", "cible"):
            o = r.get(nom)
            print(f"  [{src:6s}] {nom:7s} " + (f"x = {o['xyz'][0]:.4f}  y = {o['xyz'][1]:.4f} m" if o else "non trouvé"))
    if d435 and d435.get("info"):
        print(f"  [d435  ] {d435['info']}")
    if ecarts:
        print(f"  écarts webcam / D435 : {ecarts}")
        lacet_compte = (sc.get("posture") != "debout")      # webcam : lacet du poteau peu fiable
        if ecarts.get("planche_xy_mm", 0) > 30 or (lacet_compte and ecarts.get("planche_lacet_deg", 0) > 10) \
                or ecarts.get("posture", "identique") != "identique":
            print("  [ATTENTION] webcam et D435 en désaccord sur la planche : regarder les deux images"
                  " annotées avant d'utiliser cette scène")
    print(f"  scène retenue : {sc}")
    print("  -> perception/detection_planche.json")


# ---------------------------------------------------------------- main
def faire_d435(args, ref=None):
    """Une détection D435 : nouvelle capture (robot) ou capture enregistrée (--hors-ligne).
    ref : planche vue par la webcam (choix de l'amas D435). Recalage sur le tag et la table."""
    if args.hors_ligne:
        z = np.load(args.hors_ligne)
        prof_m, couleur, K, T_base_EE = z["prof_m"], z["couleur"], z["K"], z["T_base_EE"]
    else:
        if not args.ip:
            raise SystemExit("[ERREUR] --ip de l'Ubuntu nécessaire (pose du robot pour placer la D435)")
        print("[d435] capture (ne pas toucher le bras)...")
        prof_m, couleur, K, T_base_EE = capturer_d435(args.ip)
    T_bc = T_base_EE @ T_EE_cam()
    import recalage_d435 as rc
    T_bc, info_rc = rc.recaler(prof_m, couleur, K, T_bc)
    print(f"[d435] recalage tag + table : {info_rc}")
    print(f"[d435] caméra à z = {T_bc[2, 3]:.3f} m, axe optique (base) = {np.round(T_bc[:3, 2], 2)}")
    if T_bc[2, 2] > -0.5:
        print("[attention] la D435 ne regarde pas vers la table (pince à orienter vers le bas)")
    d435 = detecter_d435(prof_m, couleur, K, T_bc, ref=ref)
    img_d = annoter_d435(couleur, d435, K, T_bc)
    pl = d435["planche"]
    if pl and pl["posture"] == "debout" and T_bc[2, 3] - (red_z(pl["x"], pl["y"]) + HAUTEUR["debout"]) < 0.30:
        print("[attention] haut du poteau à moins de 30 cm de la D435 : profondeur peu fiable")
    return d435, img_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="image webcam BRUTE (comme vue_webcam.png)")
    ap.add_argument("--d435", action="store_true", help="D435 seule")
    ap.add_argument("--les-deux", action="store_true", help="webcam + D435 (nouvelle photo D435 à chaque ESPACE)")
    ap.add_argument("--ip", help="IP de l'Ubuntu (serveur_robot.py) pour lire O_T_EE")
    ap.add_argument("--hors-ligne", help="rejouer une capture D435 (.npz) sans robot ni caméra")
    ap.add_argument("--posture", choices=POSTURES, help="imposer la posture à la webcam")
    args = ap.parse_args()
    postures = (args.posture,) if args.posture else POSTURES
    avec_d435 = args.d435 or args.les_deux or bool(args.hors_ligne)

    if avec_d435 and not (args.les_deux or args.image):
        d435, img_d = faire_d435(args)
        return enregistrer(None, d435, None, img_d)

    from detection import Webcam, ouvrir_webcam
    cam = Webcam()
    if args.image:
        img_c = cam.corriger(cv2.imread(args.image))
        t0 = time.time()
        res, ct = detecter_webcam(img_c, cam, postures)
        print(f"[webcam] ajustement en {time.time() - t0:.1f} s")
        d435, img_d = faire_d435(args, res["planche"]) if avec_d435 else (None, None)
        return enregistrer(res, d435, annoter_webcam(img_c, res, ct, cam), img_d)

    table = masque_table(cam)
    cap = ouvrir_webcam()
    titre = "Planche (ESPACE = detecter, q = quitter)"
    print("Direct : en jaune = pixels de la couleur de la planche (pas encore la détection).")
    print("ESPACE = détecter" + (" avec la webcam ET une nouvelle photo D435" if avec_d435 else " (webcam)")
          + " ; le résultat s'affiche, une touche pour revenir au direct ; q = quitter")
    while True:
        ok, img = cap.read()
        if not ok:
            raise SystemExit("[ERREUR] lecture webcam impossible")
        img_c = cam.corriger(img)
        vue = img_c.copy()
        m = masque_planche(img_c, table)
        vue[m > 0] = (0.5 * vue[m > 0] + [0, 100, 127]).astype(np.uint8)    # aperçu du masque
        cv2.putText(vue, "Direct : en jaune = pixels couleur planche. ESPACE = detecter", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(titre, vue)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            res, ct = detecter_webcam(img_c, cam, postures)
            d435, img_d = faire_d435(args, res["planche"]) if avec_d435 else (None, None)
            ann = annoter_webcam(img_c, res, ct, cam)
            cv2.imwrite(os.path.join(HERE, "vue_webcam_planche.png"), img)
            enregistrer(res, d435, ann, img_d)
            # chaque détection est gardée (pour la rejouer hors ligne) : captures_planche/NN_*
            dos = os.path.join(HERE, "captures_planche")
            os.makedirs(dos, exist_ok=True)
            n = len([f for f in os.listdir(dos) if f.endswith("_webcam.png")]) + 1
            cv2.imwrite(os.path.join(dos, f"{n:02d}_webcam.png"), img)
            if avec_d435 and not args.hors_ligne:
                import shutil
                shutil.copy(os.path.join(HERE, "d435_capture.npz"), os.path.join(dos, f"{n:02d}_d435.npz"))
            print(f"  -> capture n°{n} gardée dans perception/captures_planche/")
            if img_d is not None:                       # webcam | D435 côte à côte
                ann = np.hstack([ann, cv2.resize(img_d, (W, H))])
                ann = cv2.resize(ann, (W, H // 2))
            cv2.putText(ann, "Resultat. Une touche pour revenir au direct", (10, ann.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(titre, ann)
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
