"""Scène MuJoCo « twin » : le Panda dans la VRAIE géométrie du setup du labo.

Ne modifie pas panda_pick_env.py (l'ancienne scène reste intacte pour l'IL/RL déjà entraînés).
Toutes les mesures viennent de la calibration du 20/11/2026 (méthode et chiffres : README, section « Digital twin on the real robot ») :
  - perception/T_base_tag0.json      : tag 0 dans le repère robot (coins touchés au crayon)
  - perception/webcam_pose_base.json : webcam dans le repère robot (tag + solvePnP)
  - tables : dimensions mesurées au mètre, bords retrouvés sur l'image webcam

Repère : celui de la base du robot (x vers la table rouge, z vers le haut, z = 0 = table blanche).

Utilisation :
    python common/twin_scene.py                  -> ouvre le viewer MuJoCo sur la scène twin
    python common/twin_scene.py --render         -> enregistre les vues virtuelles 1280x720
                                                    (perception/sim_webcam.png et sim_d435.png)
    python common/twin_scene.py --render --q "0 -0.3 0 -2.2 0 1.9 0.8"   -> avec ces 7 angles
Depuis un autre script :
    from twin_scene import build_twin_model, WEBCAM
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
PANDA_DIR = os.environ.get("PANDA_DIR", os.path.join(ROOT, "mujoco_menagerie", "franka_emika_panda"))
CALIB = os.path.join(ROOT, "perception")

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

# ---------------------------------------------------------------- mesures du setup réel
# table blanche (porte le robot) : 45 x 45 cm, dessus à z = 0
WHITE_X, WHITE_Y, WHITE_Z = (-0.25, 0.20), (-0.22, 0.24), 0.0
# table rouge (zone de travail) : 70 x 140 cm ; dessus mesuré au crayon à -4 / -4 / +1 mm.
# Largeur (x) : bords retrouvés sur l'image webcam. Longueur (y) : le bout côté trou/trépied est
# ~10 cm après le passe-câbles (trou en y = +0,31 m sur l'image) -> ESTIMATION à confirmer au mètre ;
# le reste des 1,40 m part vers le mur (côté boîtier de commande et arrêt d'urgence).
# Mesuré au crayon le 23/11 : bord côté robot à x = 0,198 m (parallèle à l'axe y à 0,1° près),
# coin côté tag/trépied à y = 0,458 m.
RED_Y_END = 0.458
RED_X, RED_Y = (0.198, 0.898), (RED_Y_END - 1.40, RED_Y_END)
# Le plateau rouge n'est PAS horizontal : plan z = a.x + b.y + c ajusté sur 9 points touchés au
# crayon (20/11 : 3 sur la table + 4 coins du tag en papier ; 23/11 : 2 sur le bord côté robot),
# résidus <= 2,7 mm. Pente ~2,1° : le bord côté robot est ~1 cm au-dessus de la table blanche.
RED_PLANE = (-0.0369, -0.0062, 0.0178)


def red_z(x, y):
    """Hauteur du dessus de la table rouge au point (x, y) du repère base."""
    a, b, c = RED_PLANE
    return a * x + b * y + c


RED_Z = red_z(0.5, 0.0)       # hauteur moyenne dans la zone de travail (~ -0,8 cm)
TABLE_THICK = 0.03            # épaisseur des plateaux (visuelle seulement)
FLOOR_Z = -0.75               # sol (approximatif, seulement pour l'image)

CUBE_HALF = 0.02              # cube 40 mm
CUBE_MASS = 0.0233            # pesé le 20/11 : 23,3 g
TARGET_R, TARGET_H = 0.030, 0.0015   # cible : disque Ø 60 x 1,5 mm (STL cible_mm.stl)
TAG_BLACK, TAG_WHITE = 0.080, 0.100  # tag 36h11 : carré noir 80 mm, marge blanche ~10 mm
# motif réel du tag36h11 ID 0 : 8 x 8 cases de 10 mm (bordure noire + 6 x 6 données), 1 = blanc.
# Ligne 0 = haut de l'image du tag = côté +y du repère tag (coins c0 -> c1 en haut), comme OpenCV.
TAG_BITS = [[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]]

# D435 au poignet — POSE NOMINALE (20/11) tirée du STL du support Franka + photos du montage,
# dans le repère "hand" (origine = bride, axes = ceux de l'EE : z vers les doigts, y = ouverture des doigts).
# Incertitude ~1-2 cm et quelques degrés (vis utilisée, position du capteur couleur dans le boîtier) :
# à remplacer par la calibration main-oeil (tag fixe vu par la D435 + O_T_EE lu sur le robot).
D435_POS = [0.045, 0.0, 0.064]       # centre optique couleur : 4,5 cm sur le côté, 6,4 cm sous la bride
D435_XYAXES = [0, 1, 0, 1, 0, 0]     # regarde vers les doigts (+z hand), haut de l'image vers l'extérieur
D435_FOVY = 43.24                    # mesuré (couleur 1280x720)
D435_BOX = [0.0125, 0.045, 0.0125]   # boîtier 25 x 90 x 25 mm (demi-tailles), visuel seulement
D435_CALIBREE = False
_p = os.path.join(CALIB, "d435_pose_hand.json")      # écrit par perception/calib_d435.py
if os.path.exists(_p):
    with open(_p) as _f:
        _mj = json.load(_f)["mujoco"]
    D435_POS, D435_XYAXES, D435_FOVY = _mj["pos"], _mj["xyaxes"], _mj["fovy"]
    D435_CALIBREE = True

# passe-câbles à côté du tag (trou dans la table rouge) : à ne jamais viser pour le cube ni la cible.
# Mesuré au crayon le 23/11 (2 cercles de 4 points, résidus 0,1 mm) : bague Ø 91 mm, trou Ø 74 mm,
# lèvre de la bague ~3,5 mm au-dessus du plateau.
HOLE_XY = (0.3190, 0.3104)
HOLE_R = 0.0455               # rayon extérieur de la bague grise
HOLE_R_IN = 0.0370            # rayon du trou : le cube de 4 cm passe dedans
HOLE_LIP = 0.0035             # hauteur de la lèvre de la bague


def hors_du_trou(xy, marge=0.03):
    """True si un objet centré en xy est à plus de `marge` du bord du passe-câbles."""
    return np.hypot(xy[0] - HOLE_XY[0], xy[1] - HOLE_XY[1]) > HOLE_R + marge

# couleurs proches du réel (pour la superposition)
RGBA_RED_TABLE = [0.85, 0.35, 0.33, 1]
RGBA_WHITE_TABLE = [0.92, 0.90, 0.85, 1]
RGBA_CUBE = [0.10, 0.45, 0.80, 1]       # PLA bleu
RGBA_TARGET = [0.05, 0.65, 0.40, 1]     # PLA vert


def _load_json(name, default):
    path = os.path.join(CALIB, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"[twin] {name} absent : valeurs par défaut du 20/11")
    return default


# valeurs de secours (calibration du 20/11) si les JSON ne sont pas dans perception/
_WEBCAM_DEFAULT = {"mujoco": {"pos": [0.5132, 0.5656, 0.8102],
                              "xyaxes": [-0.9857, 0.1679, 0.0149, -0.1285, -0.8059, 0.5779],
                              "fovy": 35.97, "resolution": [1280, 720]}}
_TAG_DEFAULT = {"T_base_tag0": [[-0.999324, -0.017118, 0.03254, 0.274124],
                                [0.016944, -0.999841, -0.005597, 0.19995],
                                [0.032631, -0.005042, 0.999455, 0.005959],
                                [0, 0, 0, 1]]}

WEBCAM = _load_json("webcam_pose_base.json", _WEBCAM_DEFAULT)["mujoco"]
T_BASE_TAG0 = np.array(_load_json("T_base_tag0.json", _TAG_DEFAULT)["T_base_tag0"])


def _mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).flatten())
    return q


def _quat_z(angle_rad):
    """Quaternion d'une rotation autour de z (le modèle Panda est en radians : pas d'euler en degrés)."""
    return [np.cos(angle_rad / 2), 0.0, 0.0, np.sin(angle_rad / 2)]


def _box(parent, name, x, y, z_top, thick, rgba):
    parent.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[np.mean(x), np.mean(y), z_top - thick / 2],
                    size=[(x[1] - x[0]) / 2, (y[1] - y[0]) / 2, thick / 2], rgba=rgba)


def build_twin_spec(cube_xy=(0.50, -0.10), target_xy=(0.50, 0.10), cube_yaw=0.0):
    """MjSpec de la scène twin (non compilé : on peut encore y ajouter des choses)."""
    spec = mujoco.MjSpec.from_file(os.path.join(PANDA_DIR, "scene.xml"))
    for key in list(spec.keys):          # la keyframe "home" ne connaît pas le cube
        spec.delete(key)
    for mat in spec.materials:
        mat.reflectance = 0.0
    floor = spec.geom("floor")
    floor.pos = [0, 0, FLOOR_Z]
    floor.material = ""                  # sol uni (le damier brouille la superposition)
    floor.rgba = [0.45, 0.45, 0.45, 1]
    # rendu hors écran à la résolution de la webcam
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 720)

    wb = spec.worldbody
    _box(wb, "table_blanche", WHITE_X, WHITE_Y, WHITE_Z, TABLE_THICK, RGBA_WHITE_TABLE)
    # table rouge inclinée selon le plan mesuré, AVEC un vrai trou (passe-câbles).
    # MuJoCo ne sait pas « percer » une boîte (collisions convexes) : le plateau est fait de
    # 4 boîtes autour d'une ouverture carrée + 4 coins coupés à 45° -> ouverture octogonale
    # inscrite dans le cercle du trou. Le cube peut tomber dedans, comme en vrai.
    a, b, _ = RED_PLANE
    n = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])        # normale du plateau
    xc, yc = np.mean(RED_X), np.mean(RED_Y)
    centre = np.array([xc, yc, red_z(xc, yc)]) - n * TABLE_THICK / 2
    ex = np.array([1.0, 0, a]) / np.linalg.norm([1.0, 0, a])            # axe x du plateau, dans le plan
    ey = np.cross(n, ex)
    R_tab = np.column_stack([ex, ey, n])
    plateau = wb.add_body(name="table_rouge", pos=centre.tolist(), quat=_mat2quat(R_tab).tolist())
    # coordonnées locales (dans le plan du plateau, origine au centre du plateau)
    hx_l = (np.array([HOLE_XY[0], HOLE_XY[1], red_z(*HOLE_XY)]) - centre) @ R_tab
    hu, hv = hx_l[0], hx_l[1]
    lx, ly = (RED_X[1] - RED_X[0]) / 2, (RED_Y[1] - RED_Y[0]) / 2
    r = HOLE_R_IN
    t = TABLE_THICK / 2

    def plaque(nom, u0, u1, v0, v1):
        if u1 - u0 > 1e-4 and v1 - v0 > 1e-4:
            plateau.add_geom(name=nom, type=mujoco.mjtGeom.mjGEOM_BOX,
                             pos=[(u0 + u1) / 2, (v0 + v1) / 2, 0], size=[(u1 - u0) / 2, (v1 - v0) / 2, t],
                             rgba=RGBA_RED_TABLE)

    plaque("table_rouge_a", -lx, hu - r, -ly, ly)            # tout ce qui est d'un côté du trou
    plaque("table_rouge_b", hu + r, lx, -ly, ly)             # de l'autre côté
    plaque("table_rouge_c", hu - r, hu + r, -ly, hv - r)     # devant le trou
    plaque("table_rouge_d", hu - r, hu + r, hv + r, ly)      # derrière le trou
    # coins de l'ouverture carrée bouchés par des prismes à 45° -> octogone
    # carré tourné de 45° centré sur chaque coin : sa face intérieure est à la distance r du centre
    a45 = r * (np.sqrt(2) - 1)
    for i, (su, sv) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        plateau.add_geom(name=f"table_rouge_coin{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                         pos=[hu + su * r, hv + sv * r, 0], size=[a45, a45, t * 0.999],
                         quat=_quat_z(np.pi / 4), rgba=RGBA_RED_TABLE)
    # bague grise autour du trou
    for k in range(16):
        ang = 2 * np.pi * k / 16
        rm = (HOLE_R + r) / 2
        plateau.add_geom(name=f"bague{k}", type=mujoco.mjtGeom.mjGEOM_BOX,     # lèvre en relief (avec collision)
                         pos=[hu + rm * np.cos(ang), hv + rm * np.sin(ang), t + HOLE_LIP / 2],
                         size=[(HOLE_R - r) / 2, HOLE_R * np.tan(np.pi / 16), HOLE_LIP / 2],
                         quat=_quat_z(ang), rgba=[0.35, 0.35, 0.37, 1])

    # tag 0 : feuille blanche + carré noir. La feuille est posée À PLAT sur la table inclinée :
    # on garde la position (x, y) et l'orientation dans le plan (lacet) mesurées, mais on prend
    # la hauteur et l'inclinaison du plateau (sinon la feuille, inclinée un peu différemment,
    # s'enfonce à moitié dans la table). Écart avec la mesure brute < 1 mm.
    yaw = np.arctan2(T_BASE_TAG0[1, 0], T_BASE_TAG0[0, 0])
    a, b, _ = RED_PLANE
    n = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    ex = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    ex -= n * (ex @ n)
    ex /= np.linalg.norm(ex)
    R_tag = np.column_stack([ex, np.cross(n, ex), n])
    tx, ty = T_BASE_TAG0[0, 3], T_BASE_TAG0[1, 3]
    tag = wb.add_body(name="tag0", pos=[tx, ty, red_z(tx, ty) + 0.0004], quat=_mat2quat(R_tag).tolist())
    tag.add_geom(name="tag_blanc", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[TAG_WHITE / 2, TAG_WHITE / 2, 0.0002], pos=[0, 0, 0.0],
                 rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
    tag.add_geom(name="tag_noir", type=mujoco.mjtGeom.mjGEOM_BOX,
                 size=[TAG_BLACK / 2, TAG_BLACK / 2, 0.0002], pos=[0, 0, 0.0003],
                 rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)
    # cases blanches du motif (identique au tag imprimé : il est détecté sur l'image simulée)
    cell = TAG_BLACK / 8
    for i, ligne in enumerate(TAG_BITS):
        for j, bit in enumerate(ligne):
            if bit:
                tag.add_geom(name=f"tag_case_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX,
                             size=[cell / 2, cell / 2, 0.0002],
                             pos=[-TAG_BLACK / 2 + (j + 0.5) * cell, TAG_BLACK / 2 - (i + 0.5) * cell, 0.0005],
                             rgba=[1, 1, 1, 1], contype=0, conaffinity=0)
    for i, (sx, sy) in enumerate([(-1, 1), (1, 1), (1, -1), (-1, -1)]):   # coins c0..c3 (ordre OpenCV)
        tag.add_site(name=f"tag_c{i}", pos=[sx * TAG_BLACK / 2, sy * TAG_BLACK / 2, 0.001],
                     size=[0.003, 0, 0], rgba=[1, 1, 0, 1])

    # cube (bleu, 23,3 g) et cible (disque vert)
    cube = wb.add_body(name="cube", pos=[cube_xy[0], cube_xy[1], red_z(*cube_xy) + CUBE_HALF + 0.002],
                       quat=_quat_z(cube_yaw))
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                  mass=CUBE_MASS, friction=[1.5, 0.01, 0.001], rgba=RGBA_CUBE)
    target = wb.add_body(name="target", mocap=True, pos=[target_xy[0], target_xy[1], red_z(*target_xy)])
    target.add_geom(name="target_mark", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[TARGET_R, TARGET_H / 2, 0], pos=[0, 0, TARGET_H / 2],
                    rgba=RGBA_TARGET, contype=0, conaffinity=0)

    # D435 au poignet (pose nominale, voir D435_POS) : caméra + boîtier visuel sans collision
    hand = spec.body("hand")
    hand.add_camera(name="d435", pos=D435_POS, xyaxes=D435_XYAXES, fovy=D435_FOVY)
    hand.add_geom(name="d435_boitier", type=mujoco.mjtGeom.mjGEOM_BOX, size=D435_BOX,
                  pos=[D435_POS[0], D435_POS[1], D435_POS[2] - D435_BOX[2] + 0.003],
                  rgba=[0.75, 0.75, 0.78, 1], contype=0, conaffinity=0)

    # webcam globale à sa pose calibrée (sténopé idéal, fovy réel)
    wb.add_camera(name="webcam", pos=WEBCAM["pos"], xyaxes=WEBCAM["xyaxes"], fovy=WEBCAM["fovy"])
    return spec


def build_twin_model(**kw):
    return build_twin_spec(**kw).compile()


def set_arm(model, data, q):
    for i in range(7):
        data.qpos[model.jnt_qposadr[model.joint(f"joint{i + 1}").id]] = q[i]
    for f in ("finger_joint1", "finger_joint2"):
        data.qpos[model.jnt_qposadr[model.joint(f).id]] = 0.04
    data.ctrl[:7] = q
    mujoco.mj_forward(model, data)


def render_webcam(model, data, w=1280, h=720, segmentation=False, camera="webcam"):
    """Image RGB d'une caméra virtuelle ('webcam' ou 'd435'), ou (segmentation=True) l'id de
    géométrie de chaque pixel."""
    r = mujoco.Renderer(model, h, w)
    if segmentation:
        r.enable_segmentation_rendering()
    r.update_scene(data, camera=camera)
    img = r.render().copy()
    r.close()
    if not segmentation:
        return img
    # une étiquette par OBJET (pas par géométrie) : les morceaux du plateau percé ou de la bague
    # ne doivent pas dessiner de frontières entre eux
    ids, types = img[:, :, 0].astype(np.int64), img[:, :, 1]
    noms = {}
    lut = np.full(model.ngeom + 1, -1, np.int64)
    for g in range(model.ngeom):
        nom = model.geom(g).name or f"g{g}"
        if nom.startswith("table_rouge"):
            nom = "table_rouge"
        if nom.startswith("tag_case"):
            nom = "tag_noir"
        nom = nom.rstrip("0123456789")
        lut[g] = noms.setdefault(nom, len(noms))
    out = np.full(ids.shape, -1, np.int64)
    sel = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0) & (ids < model.ngeom)
    out[sel] = lut[ids[sel]]
    return out


def main():
    q = HOME_Q
    if "--q" in sys.argv:
        q = np.array([float(v) for v in sys.argv[sys.argv.index("--q") + 1].split()])
    model = build_twin_model()
    data = mujoco.MjData(model)
    set_arm(model, data, q)
    if "--render" in sys.argv:
        import cv2
        out = os.path.join(CALIB, "sim_webcam.png")
        cv2.imwrite(out, render_webcam(model, data)[:, :, ::-1])
        out2 = os.path.join(CALIB, "sim_d435.png")
        cv2.imwrite(out2, render_webcam(model, data, camera="d435")[:, :, ::-1])
        print(f"Vues virtuelles -> {out} et {out2}")
        return
    viewer_fleches(model, data)


def viewer_fleches(model, data):
    """Viewer MuJoCo où les FLÈCHES gauche/droite changent de caméra (libre -> webcam -> d435).
    (Les touches [ et ] du viewer standard ne marchent pas sur un clavier AZERTY.)
    Tab / Maj+Tab : masquer les panneaux de gauche / droite si l'interface déborde de l'écran."""
    import time
    from mujoco import viewer     # (un "import mujoco.viewer" ici masquerait le module mujoco global)

    cams = [-1] + [model.camera(n).id for n in ("webcam", "d435")]
    noms = ["libre", "webcam", "d435"]
    etat = {"i": 0, "change": False}

    def touche(key):
        if key in (262, 263):                       # 262 = flèche droite, 263 = flèche gauche
            etat["i"] = (etat["i"] + (1 if key == 262 else -1)) % len(cams)
            etat["change"] = True

    print("Viewer : clic gauche = tourner, clic droit = deplacer, molette = zoom.")
    print("Fleches gauche/droite : camera libre -> webcam -> d435. Tab / Maj+Tab : afficher les panneaux.")
    # panneaux masqués au départ : avec la mise à l'échelle de Windows (125-150 %) ils débordent
    # de la fenêtre ; Tab / Maj+Tab pour les afficher
    try:
        ctx = viewer.launch_passive(model, data, key_callback=touche,
                                    show_left_ui=False, show_right_ui=False)
    except TypeError:                                # MuJoCo trop ancien : options absentes
        ctx = viewer.launch_passive(model, data, key_callback=touche)
    with ctx as v:
        while v.is_running():
            t0 = time.time()
            with v.lock():
                if etat["change"]:
                    etat["change"] = False
                    c = cams[etat["i"]]
                    if c < 0:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    else:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        v.cam.fixedcamid = c
                    print(f"camera : {noms[etat['i']]}")
                mujoco.mj_step(model, data, nstep=10)      # 10 x 2 ms = 20 ms de simulation
            v.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))   # temps réel, sans monopoliser le CPU

if __name__ == "__main__":
    main()
