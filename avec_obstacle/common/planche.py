"""Obstacle du jumeau : la planche d'aggloméré du labo (pesée et mesurée le 24/11/2026).

    53,58 x 10 x 3 cm, 1022,2 g (densité 0,64 g/cm3 : panneau de particules)

Trois postures, posée sur la table rouge INCLINÉE (plan red_z), lacet quelconque :
    "plat"    : couchée sur sa grande face   -> 3 cm de haut
    "tranche" : sur sa tranche longue        -> 10 cm de haut (un muret)
    "debout"  : sur son petit bout (10 x 3)  -> 53,58 cm de haut (un poteau)

La planche est un obstacle FIXE dans la sim (corps « mocap ») : en vrai elle tient debout sans
problème (constaté le 24/11), alors que dans MuJoCo un poteau de 54 cm sur une base de 3 cm finit
parfois par basculer à cause de la souplesse des contacts. Comme TOUT contact robot / planche est un
échec (épisode arrêté), sa dynamique ne sert à rien : on la fige.

Ce fichier ne dépend que de twin_scene.py :
    ajouter_planche(spec)            -> ajoute le corps "planche" (mocap) à la scène
    pose_planche(posture, x, y, lacet) -> position (3) + quaternion (4) de la planche
    tirer_scene(rng)                 -> cube, cible, planche tirés au hasard (contraintes du setup réel)
    empreinte(scene)                 -> les 4 coins de la planche au sol (pour les distances)
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))   # twin_scene (scène du labo)
from twin_scene import (HOLE_R, HOLE_XY, RED_PLANE, RED_X, RED_Y, T_BASE_TAG0,
                        TAG_WHITE, red_z)

# ---------------------------------------------------------------- mesures de la vraie planche
PLANCHE_L, PLANCHE_LARG, PLANCHE_EP = 0.5358, 0.10, 0.03      # m
PLANCHE_MASSE = 1.0222                                        # kg
PLANCHE_FROTTEMENT = 0.6                                      # aggloméré brut sur stratifié (estimation)
RGBA_PLANCHE = [0.95, 0.75, 0.10, 1]     # jaune : couleur du scotch prévu pour la détection webcam

POSTURES = ("plat", "tranche", "debout")
# hauteur (dimension verticale) et empreinte au sol (longueur, largeur) de chaque posture
HAUTEUR = {"plat": PLANCHE_EP, "tranche": PLANCHE_LARG, "debout": PLANCHE_L}
EMPREINTE = {"plat": (PLANCHE_L, PLANCHE_LARG), "tranche": (PLANCHE_L, PLANCHE_EP),
             "debout": (PLANCHE_LARG, PLANCHE_EP)}

# zone de travail du cube et de la cible (même zone que planif_expert.py --test)
ZONE_OBJETS = ((0.40, 0.68), (-0.25, 0.25))
DIST_CUBE_CIBLE_MIN = 0.15
# marges (m) entre le BORD de la planche et le centre du cube / de la cible : la pince ouverte
# (8 cm) doit pouvoir descendre autour du cube sans toucher la planche
MARGE_OBJET = 0.075
MARGE_BASE = 0.30          # planche à plus de 30 cm de l'axe du robot (le bras doit pouvoir passer)
MARGE_BORD_TABLE = 0.02
# proportion des scènes où la planche est mise SUR le chemin cube -> cible (le reste : n'importe où,
# souvent hors du chemin : la bonne politique va alors tout droit)
P_SUR_CHEMIN = 0.65


# ---------------------------------------------------------------- géométrie
def _normale_table():
    a, b, _ = RED_PLANE
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


def _quat_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).flatten())
    return q


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# rotation « posture » : la boîte est définie avec x = longueur, y = largeur, z = épaisseur
_R_POSTURE = {
    "plat": np.eye(3),
    "tranche": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]]),   # largeur (10 cm) verticale
    "debout": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]]),     # longueur (53,6 cm) verticale,
                                                                 # largeur (10 cm) selon le lacet
}


def _rot_vers_normale(n):
    """Rotation qui amène z sur la normale n de la table (petite inclinaison, ~2°)."""
    z = np.array([0, 0, 1.0])
    v = np.cross(z, n)
    s, c = np.linalg.norm(v), z @ n
    if s < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def pose_planche(posture, x, y, lacet):
    """Position + quaternion (7 valeurs) de la planche posée sur la table rouge inclinée.
    Lacet = angle de la grande dimension AU SOL (longueur pour plat/tranche, largeur pour debout)."""
    n = _normale_table()
    R = _rot_vers_normale(n) @ _rot_z(lacet) @ _R_POSTURE[posture]
    centre = np.array([x, y, red_z(x, y)]) + n * (HAUTEUR[posture] / 2 + 0.0005)
    return np.r_[centre, _quat_mat(R)]


def empreinte(posture, x, y, lacet):
    """4 coins (x, y) de la planche au sol."""
    L, l = EMPREINTE[posture]
    c, s = np.cos(lacet), np.sin(lacet)
    u, v = np.array([c, s]), np.array([-s, c])
    return np.array([[x, y] + su * L / 2 * u + sv * l / 2 * v
                     for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def dist_point_rectangle(p, posture, x, y, lacet):
    """Distance (m) d'un point (x, y) au rectangle de l'empreinte (0 si dedans)."""
    L, l = EMPREINTE[posture]
    d = np.asarray(p, float) - [x, y]
    c, s = np.cos(lacet), np.sin(lacet)
    du, dv = abs(d @ [c, s]), abs(d @ [-s, c])
    return float(np.hypot(max(du - L / 2, 0), max(dv - l / 2, 0)))


# ---------------------------------------------------------------- modèle
def ajouter_planche(spec):
    """Ajoute la planche (boîte 53,58 x 10 x 3 cm, fixe : corps mocap) à la scène twin."""
    b = spec.worldbody.add_body(name="planche", mocap=True, pos=[0.6, 0.0, -0.5])
    b.add_geom(name="planche_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
               size=[PLANCHE_L / 2, PLANCHE_LARG / 2, PLANCHE_EP / 2], mass=PLANCHE_MASSE,
               friction=[PLANCHE_FROTTEMENT, 0.005, 0.0001], rgba=RGBA_PLANCHE)
    return b


# ---------------------------------------------------------------- tirage d'une scène
def _sur_la_table(coins):
    xs, ys = coins[:, 0], coins[:, 1]
    return (xs.min() > RED_X[0] + MARGE_BORD_TABLE and xs.max() < RED_X[1] - MARGE_BORD_TABLE
            and ys.min() > RED_Y[0] + MARGE_BORD_TABLE and ys.max() < RED_Y[1] - MARGE_BORD_TABLE)


def scene_valide(sc):
    """Contraintes du setup réel (sans la visibilité webcam, testée dans l'env)."""
    c, t = np.array(sc["cube_xy"]), np.array(sc["cible_xy"])
    po, px, py, pl = sc["posture"], sc["planche_xy"][0], sc["planche_xy"][1], sc["planche_lacet"]
    if np.linalg.norm(c - t) < DIST_CUBE_CIBLE_MIN:
        return False
    for xy in (c, t):
        if np.hypot(*(xy - HOLE_XY)) < HOLE_R + 0.03:
            return False
        if dist_point_rectangle(xy, po, px, py, pl) < MARGE_OBJET:
            return False
    coins = empreinte(po, px, py, pl)
    if not _sur_la_table(coins):
        return False
    if dist_point_rectangle((0, 0), po, px, py, pl) < MARGE_BASE:
        return False
    if dist_point_rectangle(HOLE_XY, po, px, py, pl) < HOLE_R + 0.01:      # pas sur le passe-câbles
        return False
    tag = T_BASE_TAG0[:2, 3]
    if dist_point_rectangle(tag, po, px, py, pl) < TAG_WHITE / 2 * 1.5:   # pas sur le tag de recalage
        return False
    return True


def tirer_scene(rng, posture=None, sur_chemin=None):
    """Tire cube (x, y, lacet), cible (x, y) et planche (posture, x, y, lacet) au hasard.
    posture / sur_chemin : imposer la posture ou le placement (sinon tirés au hasard)."""
    for _ in range(10000):
        c = rng.uniform([ZONE_OBJETS[0][0], ZONE_OBJETS[1][0]], [ZONE_OBJETS[0][1], ZONE_OBJETS[1][1]])
        t = rng.uniform([ZONE_OBJETS[0][0], ZONE_OBJETS[1][0]], [ZONE_OBJETS[0][1], ZONE_OBJETS[1][1]])
        if np.linalg.norm(c - t) < DIST_CUBE_CIBLE_MIN:
            continue
        po = posture or POSTURES[rng.integers(3)]
        chemin = (rng.random() < P_SUR_CHEMIN) if sur_chemin is None else sur_chemin
        if chemin:
            # planche au milieu du segment cube -> cible, plutôt EN TRAVERS (pour couper le chemin)
            s = rng.uniform(0.35, 0.65)
            p = c + s * (t - c) + rng.normal(0, 0.02, 2)
            a_chemin = np.arctan2(*(t - c)[::-1])
            lacet = a_chemin + np.pi / 2 + rng.normal(0, 0.35)
        else:
            p = rng.uniform([0.30, -0.45], [0.85, 0.40])
            lacet = rng.uniform(0, np.pi)
        sc = {"cube_xy": c.tolist(), "cube_lacet_deg": float(rng.uniform(0, 90)), "cible_xy": t.tolist(),
              "posture": po, "planche_xy": p.tolist(), "planche_lacet": float(lacet % np.pi),
              "sur_chemin": bool(chemin)}
        if scene_valide(sc):
            return sc
    raise RuntimeError("impossible de tirer une scène valide")
