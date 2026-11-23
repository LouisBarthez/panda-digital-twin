"""Détection du cube (bleu) et de la cible (verte) sur la webcam globale -> positions dans le repère robot.

Méthode (piste A du plan sim2real) :
  1. image remappée sur la caméra idéale (distorsion corrigée, même convention que la scène twin) ;
  2. seuillage HSV : bleu = cube, vert = cible ; on garde le plus gros objet de chaque couleur
     dont la position tombe sur la table rouge (pas le scotch bleu du robot, pas un objet hors table) ;
  3. centre de l'objet dans l'image -> rayon lumineux depuis la webcam (pose calibrée) ->
     intersection avec le plan de la table (incliné, mesuré au crayon), relevé de la demi-hauteur
     du cube (le centre de la silhouette d'un cube vu en biais ≈ projection de son centre 3D) ;
  4. orientation du cube (lacet) par le rectangle minimal de sa silhouette.

Usage (PowerShell, venv activé, à la racine du dépôt) :
    python perception/detection.py                  -> webcam en direct ; ESPACE = enregistrer, q = quitter
    python perception/detection.py --image perception/vue_webcam.png
    python perception/detection.py --reglage        -> curseurs pour régler les seuils HSV en direct
    --brut : sans la correction fine de correction_detection.json (apprise par corriger_detection.py)
    --centroide : ancienne méthode pour le cube (centre de la tache + rectangle minimal) au lieu de
                  l'ajustement d'un cube 3D sur la silhouette (méthode « modele », par défaut)
Mesure de l'erreur : en direct, après ESPACE, donner une étiquette d'essai (e1, e2...) ; puis toucher au
crayon le centre du dessus du cube et le centre de la cible (lire_pose, étiquettes cube_e1, cible_e1...)
et lancer perception/erreur_detection.py.
Sorties : perception/detection.json (positions, lues par superposition.py pour placer cube et cible
dans la scène), perception/detection.png (image annotée).
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from webcam_index import index_webcam  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "common"))
from apriltag_pose import load_calibration  # noqa: E402
from twin_scene import (CUBE_HALF, HOLE_R, HOLE_XY, RED_PLANE, RED_X, RED_Y,  # noqa: E402
                        T_BASE_TAG0, TARGET_H, WEBCAM, red_z)

# méthode pour le cube : "modele" (ajustement d'un cube 3D sur la silhouette, par défaut)
# ou "centroide" (centre de la tache + rectangle minimal, 1re version) avec --centroide
METHODE = "centroide" if "--centroide" in sys.argv else "modele"

W, H = 1280, 720
SEUILS_FICHIER = os.path.join(HERE, "seuils_hsv.json")
# seuils HSV OpenCV (H : 0-180, S et V : 0-255) — réglables avec --reglage, sauvés dans seuils_hsv.json
SEUILS_DEFAUT = {"cube": {"bas": [95, 120, 40], "haut": [130, 255, 255]},
                 "cible": {"bas": [55, 80, 40], "haut": [92, 255, 255]}}
AIRE_MIN = 150          # pixels : en dessous, on ignore (bruit)


def charger_seuils():
    if os.path.exists(SEUILS_FICHIER):
        with open(SEUILS_FICHIER) as f:
            return json.load(f)
    return json.loads(json.dumps(SEUILS_DEFAUT))


class Webcam:
    """Remappage vers la caméra idéale + passage pixel -> point du repère robot."""

    def __init__(self):
        K, D, _ = load_calibration(os.path.join(HERE, "webcam_intrinsics.json"))
        f = (H / 2) / np.tan(np.radians(WEBCAM["fovy"]) / 2)
        self.Ki = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        self.m1, self.m2 = cv2.initUndistortRectifyMap(K, D, None, self.Ki, (W, H), cv2.CV_32FC1)
        with open(os.path.join(HERE, "webcam_pose_base.json")) as fp:
            self.T_bc = np.array(json.load(fp)["T_base_cam"])

    def corriger(self, img):
        return cv2.remap(img, self.m1, self.m2, cv2.INTER_LINEAR)

    def pixel_vers_plan(self, u, v, hauteur):
        """Point du plan « table + hauteur » vu au pixel (u, v) de l'image corrigée."""
        ray_c = np.array([(u - self.Ki[0, 2]) / self.Ki[0, 0], (v - self.Ki[1, 2]) / self.Ki[1, 1], 1.0])
        d = self.T_bc[:3, :3] @ ray_c
        o = self.T_bc[:3, 3]
        a, b, c = RED_PLANE
        # o + t.d sur le plan z = a.x + b.y + c + hauteur
        t = (a * o[0] + b * o[1] + c + hauteur - o[2]) / (d[2] - a * d[0] - b * d[1])
        return o + t * d

    def point_vers_pixel(self, p):
        q = np.linalg.inv(self.T_bc) @ np.r_[p, 1.0]
        return self.Ki[0, 0] * q[0] / q[2] + self.Ki[0, 2], self.Ki[1, 1] * q[1] / q[2] + self.Ki[1, 2]


def sur_la_table(p):
    if not (RED_X[0] + 0.01 < p[0] < RED_X[1] - 0.01 and RED_Y[0] + 0.01 < p[1] < RED_Y[1] - 0.01):
        return False
    if np.hypot(p[0] - HOLE_XY[0], p[1] - HOLE_XY[1]) < HOLE_R:        # dans le trou
        return False
    if np.hypot(p[0] - T_BASE_TAG0[0, 3], p[1] - T_BASE_TAG0[1, 3]) < 0.06:   # sur le tag
        return False
    return True


def charger_correction():
    """Corrections affines apprises au robot, par objet : {'cube': M ou None, 'cible': M ou None}.
    La correction du cube n'est appliquée que si elle a été apprise avec la même méthode."""
    path = os.path.join(HERE, "correction_detection.json")
    if "--brut" in sys.argv or not os.path.exists(path):
        return {"cube": None, "cible": None}
    with open(path) as f:
        c = json.load(f)
    if "M" in c:                                        # ancien format : une correction commune
        c = {"cube": {"M": c["M"], "methode_detection": c.get("methode_detection", "centroide")},
             "cible": {"M": c["M"]}}
    out = {"cible": np.array(c["cible"]["M"]) if c.get("cible") else None, "cube": None}
    cc = c.get("cube")
    if cc:
        if cc.get("methode_detection", "centroide") == METHODE:
            out["cube"] = np.array(cc["M"])
        else:
            print(f"[info] correction du cube apprise avec la méthode '{cc.get('methode_detection')}' :"
                  f" non appliquée à la méthode '{METHODE}' (voir evaluer_detection.py)")
    return out


CORRECTION = charger_correction()


def silhouette_cube(cam, x, y, lacet, ox, oy):
    """Contour (pixels, relatifs à ox, oy) d'un cube de 40 mm posé en (x, y) avec ce lacet."""
    z0 = red_z(x, y)
    c, s_ = np.cos(lacet), np.sin(lacet)
    h = CUBE_HALF
    pts = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        px, py = x + c * sx * h - s_ * sy * h, y + s_ * sx * h + c * sy * h
        for z in (z0, z0 + 2 * h):
            u, v = cam.point_vers_pixel(np.array([px, py, z]))
            pts.append([u - ox, v - oy])
    return cv2.convexHull(np.array(pts, np.float32)).astype(np.int32)


def raffiner_masque_cube(img_c, contour):
    """Masque plus fidèle du cube autour de la tache HSV : le seuil HSV rogne les bords flous et
    la face sombre. Ici on utilise « bleu - rouge » (le cube est bleu, la table rouge : contraste
    maximal, faces sombres comprises) avec un seuil automatique d'Otsu calculé localement."""
    bx, by, bw, bh = cv2.boundingRect(contour)
    m = 25
    H_, W_ = img_c.shape[:2]
    x0, y0, x1, y1 = max(bx - m, 0), max(by - m, 0), min(bx + bw + m, W_), min(by + bh + m, H_)
    roi = img_c[y0:y1, x0:x1].astype(np.int16)
    dif = np.clip(roi[..., 0] - roi[..., 2] + 128, 0, 255).astype(np.uint8)
    _, mk = cv2.threshold(cv2.GaussianBlur(dif, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return contour
    mo = cv2.moments(contour)
    cu, cv_ = mo["m10"] / mo["m00"] - x0, mo["m01"] / mo["m00"] - y0
    dedans = [c for c in cts if cv2.pointPolygonTest(c, (cu, cv_), False) >= 0]
    best = max(dedans or cts, key=cv2.contourArea)
    # garde-fou : le masque affiné ne doit pas exploser (ombre, autre objet bleu collé)
    if cv2.contourArea(best) > 2.5 * cv2.contourArea(contour):
        return contour
    return best + np.array([x0, y0])


def ajuster_cube(cam, contour, x0, y0):
    """Ajustement par modèle (« analyse par synthèse ») : on cherche la position (x, y) et le lacet
    du cube 3D dont la silhouette projetée recouvre le mieux la tache bleue détectée (IoU).
    Grille grossière (lacet tous les 3°, ±8 mm) puis fine (0,5°, 1 mm)."""
    bx, by, bw, bh = cv2.boundingRect(contour)
    m = 40
    ox, oy = bx - m, by - m
    W_, H_ = bw + 2 * m, bh + 2 * m
    cible = np.zeros((H_, W_), np.uint8)
    cv2.drawContours(cible, [contour - [ox, oy]], -1, 1, -1)
    n_c = cible.sum()
    buf = np.zeros_like(cible)

    def score(x, y, lac):
        buf[:] = 0
        cv2.fillConvexPoly(buf, silhouette_cube(cam, x, y, lac, ox, oy), 1)
        inter = np.count_nonzero(buf & cible)
        return inter / (n_c + np.count_nonzero(buf) - inter)

    best = (-1.0, x0, y0, 0.0)
    for lac in np.radians(np.arange(0, 90, 3)):
        for dx in (-0.008, -0.004, 0, 0.004, 0.008):
            for dy in (-0.008, -0.004, 0, 0.004, 0.008):
                sc = score(x0 + dx, y0 + dy, lac)
                if sc > best[0]:
                    best = (sc, x0 + dx, y0 + dy, lac)
    _, xb, yb, lb = best
    for lac in lb + np.radians(np.arange(-3, 3.01, 0.5)):
        for dx in np.arange(-0.003, 0.0031, 0.001):
            for dy in np.arange(-0.003, 0.0031, 0.001):
                sc = score(xb + dx, yb + dy, lac)
                if sc > best[0]:
                    best = (sc, xb + dx, yb + dy, lac)
    sc, x, y, lac = best
    return x, y, float(np.degrees(lac) % 90), sc


def detecter(img_c, cam, seuils):
    """Renvoie {'cube': {...} ou None, 'cible': {...} ou None} et les masques."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img_c, (5, 5), 0), cv2.COLOR_BGR2HSV)
    res, masques = {}, {}
    for nom, hauteur in (("cube", CUBE_HALF), ("cible", TARGET_H)):
        m = cv2.inRange(hsv, np.array(seuils[nom]["bas"]), np.array(seuils[nom]["haut"]))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        masques[nom] = m
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        meilleur = None
        for ct in sorted(contours, key=cv2.contourArea, reverse=True):
            aire = cv2.contourArea(ct)
            if aire < AIRE_MIN:
                break
            mo = cv2.moments(ct)
            u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
            p = cam.pixel_vers_plan(u, v, hauteur)
            if not sur_la_table(p):
                continue
            lacet = None
            iou = None
            if nom == "cube" and METHODE == "modele":
                ct = raffiner_masque_cube(img_c, ct)
                x, y, lacet, iou = ajuster_cube(cam, ct, p[0], p[1])
                p = np.array([x, y, red_z(x, y) + CUBE_HALF])
            elif nom == "cube":
                # lacet : rectangle minimal des points du contour reportés sur le plan du dessus du cube
                pts = np.array([cam.pixel_vers_plan(x, y, 2 * CUBE_HALF)[:2]
                                for x, y in cv2.convexHull(ct).reshape(-1, 2)], np.float32)
                (_, _), (_, _), ang = cv2.minAreaRect(pts)
                lacet = float(ang % 90)
            if CORRECTION.get(nom) is not None:              # correction fine apprise au robot
                p = np.r_[np.r_[p[:2], 1.0] @ CORRECTION[nom], p[2]]
            meilleur = {"pixel": [round(u, 1), round(v, 1)], "aire_px": int(aire),
                        "xyz": [round(float(x), 4) for x in p], "contour": ct, "methode": METHODE}
            if lacet is not None:
                meilleur["lacet_deg"] = round(lacet, 1)
            if iou is not None:
                meilleur["iou"] = round(iou, 3)
            break
        res[nom] = meilleur
    return res, masques


def annoter(img_c, res, cam):
    out = img_c.copy()
    for nom, col in (("cube", (255, 120, 0)), ("cible", (0, 200, 0))):
        r = res.get(nom)
        if r is None:
            cv2.putText(out, f"{nom} : non trouve", (10, 30 if nom == "cube" else 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue
        cv2.drawContours(out, [r["contour"]], -1, col, 2)
        if nom == "cube" and r.get("methode") == "modele":
            x, y = r["xyz"][:2]
            sil = silhouette_cube(cam, x, y, np.radians(r["lacet_deg"]), 0, 0)
            cv2.polylines(out, [sil], True, (255, 255, 255), 1)
        u, v = map(int, r["pixel"])
        cv2.drawMarker(out, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        x, y, z = r["xyz"]
        txt = f"{nom} : x={x:.3f} y={y:.3f} m" + (f"  lacet {r['lacet_deg']:.0f} deg" if "lacet_deg" in r else "")
        cv2.putText(out, txt, (10, 30 if nom == "cube" else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out


def enregistrer(res, img_annot):
    propre = {k: ({kk: vv for kk, vv in v.items() if kk != "contour"} if v else None) for k, v in res.items()}
    with open(os.path.join(HERE, "detection.json"), "w") as f:
        json.dump(propre, f, indent=2)
    cv2.imwrite(os.path.join(HERE, "detection.png"), img_annot)
    for k, v in propre.items():
        print(f"  {k:5s} : " + (f"x = {v['xyz'][0]:.4f}  y = {v['xyz'][1]:.4f}  z = {v['xyz'][2]:.4f} m"
                                 + (f"   lacet {v['lacet_deg']} deg" if 'lacet_deg' in v else "")
                                 if v else "non trouvé"))
    print("  -> perception/detection.json et detection.png")


def ouvrir_webcam():
    cap = cv2.VideoCapture(index_webcam(), cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    for _ in range(20):
        cap.read()
    return cap


def mode_reglage(cam):
    seuils = charger_seuils()
    cap = ouvrir_webcam()
    win = "Reglage HSV (s = sauver, q = quitter)"
    cv2.namedWindow(win)
    noms = []
    for obj in ("cube", "cible"):
        for borne in ("bas", "haut"):
            for i, canal in enumerate("HSV"):
                n = f"{obj} {canal} {borne}"
                noms.append((n, obj, borne, i))
                cv2.createTrackbar(n, win, seuils[obj][borne][i], 180 if canal == "H" else 255, lambda _: None)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for n, obj, borne, i in noms:
            seuils[obj][borne][i] = cv2.getTrackbarPos(n, win)
        img_c = cam.corriger(img)
        res, masques = detecter(img_c, cam, seuils)
        vue = annoter(img_c, res, cam)
        mini = [cv2.resize(cv2.cvtColor(masques[k], cv2.COLOR_GRAY2BGR), (W // 4, H // 4)) for k in ("cube", "cible")]
        vue[0:H // 4, W - W // 4:] = mini[0]
        vue[H // 4:H // 2, W - W // 4:] = mini[1]
        cv2.imshow(win, cv2.resize(vue, (W * 3 // 4, H * 3 // 4)))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            with open(SEUILS_FICHIER, "w") as f:
                json.dump(seuils, f, indent=2)
            print(f"seuils sauvés dans {SEUILS_FICHIER}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    cam = Webcam()
    if "--reglage" in sys.argv:
        return mode_reglage(cam)
    seuils = charger_seuils()
    if "--image" in sys.argv:
        img = cv2.imread(sys.argv[sys.argv.index("--image") + 1])
        img_c = cam.corriger(img)
        res, _ = detecter(img_c, cam, seuils)
        enregistrer(res, annoter(img_c, res, cam))
        return
    cap = ouvrir_webcam()
    print("ESPACE = enregistrer la détection, q = quitter")
    while True:
        ok, img = cap.read()
        if not ok:
            print("[ERREUR] lecture webcam impossible")
            break
        img_c = cam.corriger(img)
        res, _ = detecter(img_c, cam, seuils)
        vue = annoter(img_c, res, cam)
        cv2.imshow("Detection cube / cible (ESPACE = enregistrer, q = quitter)", vue)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            enregistrer(res, vue)
            essai = input("  Étiquette d'essai pour mesurer l'erreur (ex. e1), ou Entrée pour aucune : ").strip()
            if essai:
                f_ess = os.path.join(HERE, "detection_essais.json")
                ess = json.load(open(f_ess)) if os.path.exists(f_ess) else {}
                ess[essai] = {k2: ({kk: vv for kk, vv in v.items() if kk != "contour"} if v else None)
                              for k2, v in res.items()}
                with open(f_ess, "w") as f:
                    json.dump(ess, f, indent=1)
                os.makedirs(os.path.join(HERE, "essais_img"), exist_ok=True)
                cv2.imwrite(os.path.join(HERE, "essais_img", f"{essai}.png"), img)   # image brute
                print(f"  -> essai {essai} enregistré ({len(ess)} au total) dans detection_essais.json")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
