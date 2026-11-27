"""Jumeau numérique en direct : webcam -> scène MuJoCo -> expert simulé -> vrai robot.

  1. La webcam détecte le cube et la cible en continu ; ils se replacent dans la scène twin.
     Si le robot est connecté, le bras virtuel prend la pose du vrai bras (lue ~1 fois/s).
  2. Touche E : on fige la scène, l'expert planifie et JOUE le pick-and-place dans MuJoCo
     (collisions, prise, dépose vérifiées).
  3. Touche R (seulement si la simulation est OK et que rien n'a bougé) : le vrai robot rejoue
     EXACTEMENT les mêmes angles (serveur_robot.py sur l'Ubuntu). Pendant le mouvement, le bras
     virtuel suit le vrai en direct (angles reçus toutes les 40 ms). À la fin, le bras revient à sa
     pose de départ et la webcam mesure où est réellement le cube -> jumeau/essais_reels.json.

AVEC LA PLANCHE (option --planche, 26/11) : la planche est suivie EN CONTINU (webcam ; + D435 au
poignet avec --d435, recalée à chaque photo sur le tag 0 et la table) et placée dans la scène dès
qu'elle est stable. Touche O : le bras monte en pose « perchée » (vue d'ensemble pour la D435, tag
visible), après vérification du trajet dans le jumeau puis R. E fait choisir le trajet par la
politique (--politique il : réseau de l'IL + vérification dans le jumeau, planificateur complet en
secours ; --politique expert : planificateur complet), avec retour à la pose de départ, puis le
rejoue dans MuJoCo (tout contact avec la planche = refus). R vérifie en plus que la planche n'a pas
bougé depuis E.

Touches, dans la fenêtre « Jumeau » (cliquer dessus d'abord) :
    E = simuler   R = exécuter sur le robot   L = revenir à la vue en direct
    O = aller en pose perchée d'observation (avec --planche --d435), puis R pour y aller vraiment
    V = démarrer / arrêter l'enregistrement vidéo (réel | jumeau côte à côte -> jumeau/videos/)
    C = changer la vue du jumeau dans la vidéo (vue webcam -> vue 3/4 -> D435)
    Q / Échap = quitter
Dans le viewer MuJoCo : flèches gauche/droite = caméra libre -> webcam -> d435.

Usage (PowerShell, venv activé, à la racine du dépôt) :
    python jumeau/jumeau_live.py --robot 10.254.79.65          -> avec le robot
    python jumeau/jumeau_live.py                               -> simulation seule (R désactivé)
    python jumeau/jumeau_live.py --image perception/vue_webcam.png   -> test sans webcam
    python jumeau/jumeau_live.py --robot 10.254.79.65 --planche          -> avec la planche (IL)
    python jumeau/jumeau_live.py --robot 10.254.79.65 --planche --d435   -> planche mesurée aussi par la D435
Options : --vitesse 0.1 (facteur de vitesse libfranka, 0.02 à 0.3)   --accel 2 (rejeu sim 2x plus vite)
          --politique il|expert (avec --planche)
"""
import collections
import datetime
import json
import os
import sys
import threading
import time

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "perception"))
import detection as det  # noqa: E402
import lien_robot  # noqa: E402
from planif_expert import CUBE_HALF, HOME_Q, TARGET_H, Jumeau, planifier, red_z, rejouer, vers_json  # noqa: E402

N_STABLE = 5                 # détections consécutives pour déclarer la scène stable
# stabilité : écarts max autour de la médiane des 5 dernières détections. Le lacet du cube fluctue
# de quelques degrés d'une image à l'autre (ajustement de modèle, cube vu de loin ou vers 45-55°) :
# 6° reste très en dessous de la tolérance de la prise (~10-15°, cube 40 mm dans une pince de 80 mm).
TOL_POS, TOL_LACET = 0.004, 6.0
TOL_CHANGE_POS, TOL_CHANGE_LACET = 0.006, 8.0   # scène « modifiée » depuis la simulation
TOL_Q_DEPART = 0.03          # rad : le robot doit être là où la simulation l'a pris
ECART_FK_MAX = 10.0          # mm : au-delà, modèle sim et EE de Desk ne collent pas -> R bloqué
IOU_PLANCHE_MIN = 0.7        # R : la tache de la planche doit recouvrir celle vue au moment de E
FPS_VIDEO = 25
VW, VH = 960, 540            # taille de chaque moitié de la vidéo
BANDEAU_VIDEO = 100          # 540 + 100 = 640 : dimensions multiples de 16 pour le H.264


def arg(nom, defaut=None, type_=str):
    if nom in sys.argv:
        return type_(sys.argv[sys.argv.index(nom) + 1])
    return defaut


def ecart_lacet(a, b):
    return (a - b + 45) % 90 - 45


BLANC, VERT, ROUGE, JAUNE, GRIS = (255, 255, 255), (80, 230, 80), (80, 80, 255), (60, 220, 240), (170, 170, 170)


# ------------------------------------------------------------------------------- webcam + détection
class WebcamRapide(det.Webcam):
    """Même chose que detection.Webcam, avec l'inverse de la pose mis en cache (2x plus rapide)."""

    def __init__(self):
        super().__init__()
        self.T_cb = np.linalg.inv(self.T_bc)

    def point_vers_pixel(self, p):
        q = self.T_cb @ np.r_[p, 1.0]
        return self.Ki[0, 0] * q[0] / q[2] + self.Ki[0, 2], self.Ki[1, 1] * q[1] / q[2] + self.Ki[1, 2]


class Detecteur:
    """Deux threads : la capture garde l'image la plus récente (~30 img/s, vidéo fluide) ;
    la détection (~6 par seconde) travaille sur la dernière image disponible."""

    def __init__(self, image=None):
        self.image = image
        self.cam = WebcamRapide()
        self.seuils = det.charger_seuils()
        self.verrou = threading.Lock()
        self.img = None              # (t, image corrigée)
        self.res = None              # (t, résultats avec contours, résultats sans contours)
        self.erreur = None
        self.actif = True
        threading.Thread(target=self._capture, daemon=True).start()
        threading.Thread(target=self._detection, daemon=True).start()

    def _capture(self):
        if self.image:
            img0 = cv2.imread(self.image)
            if img0 is None:
                self.erreur = f"image introuvable : {self.image}"
                return
            img0 = self.cam.corriger(img0)
            while self.actif:
                with self.verrou:
                    self.img = (time.time(), img0)
                time.sleep(0.033)
            return
        cap = det.ouvrir_webcam()
        while self.actif:
            ok, img = cap.read()
            if not ok:
                self.erreur = "lecture webcam impossible (appli Caméra ouverte ? webcam_index.py ?)"
                time.sleep(0.5)
                continue
            img_c = self.cam.corriger(img)
            with self.verrou:
                self.img = (time.time(), img_c)
        cap.release()

    def _detection(self):
        t_fait = 0.0
        while self.actif:
            with self.verrou:
                im = self.img
            if im is None or im[0] <= t_fait:
                time.sleep(0.01)
                continue
            t_fait = im[0]
            res, _ = det.detecter(im[1], self.cam, self.seuils)
            propre = {k: ({kk: vv for kk, vv in v.items() if kk != "contour"} if v else None)
                      for k, v in res.items()}
            with self.verrou:
                self.res = (im[0], res, propre)

    def lire(self):
        """(t_image, image corrigée annotée avec la dernière détection, t_detection, résultats) ou None."""
        with self.verrou:
            im, res = self.img, self.res
        if im is None:
            return None
        if res is None:
            return im[0], im[1].copy(), None, None
        return im[0], det.annoter(im[1], res[1], self.cam), res[0], res[2]


class Stabilite:
    """Garde les N dernières détections ; donne une valeur filtrée et dit si la scène est stable."""

    def __init__(self):
        self.h = collections.deque(maxlen=N_STABLE)
        self.t_dernier = 0.0

    def ajouter(self, t, res):
        if t is None or t <= self.t_dernier:
            return
        self.t_dernier = t
        c, g = res.get("cube"), res.get("cible")
        if c is None or g is None:
            self.h.clear()
            return
        self.h.append((t, np.array(c["xyz"][:2]), float(c.get("lacet_deg", 0.0)), np.array(g["xyz"][:2])))

    def valeur(self):
        """(cube_xy, lacet, cible_xy, stable, t_premier, diagnostic) ou None."""
        if not self.h:
            return None
        cubes = np.array([e[1] for e in self.h])
        cibles = np.array([e[3] for e in self.h])
        lac = np.radians([e[2] for e in self.h]) * 4          # moyenne circulaire à 90° près
        lacet = float(np.degrees(np.arctan2(np.sin(lac).mean(), np.cos(lac).mean())) / 4 % 90)
        c, g = np.median(cubes, axis=0), np.median(cibles, axis=0)
        dc = np.linalg.norm(cubes - c, axis=1).max() * 1000
        dg = np.linalg.norm(cibles - g, axis=1).max() * 1000
        dl = max(abs(ecart_lacet(e[2], lacet)) for e in self.h)
        stable = (len(self.h) == N_STABLE and dc < TOL_POS * 1000 and dg < TOL_POS * 1000 and dl < TOL_LACET)
        diag = (f"{len(self.h)}/{N_STABLE} detections, cube +-{dc:.1f} mm, cible +-{dg:.1f} mm, lacet +-{dl:.1f} deg"
                f" (max {TOL_POS * 1000:.0f} mm / {TOL_LACET:.0f} deg)")
        return c, lacet, g, stable, self.h[0][0], diag


# ------------------------------------------------------------------------------- robot (thread)
class EtatRobot(threading.Thread):
    """Lit la pose du vrai robot ~1 fois par seconde (sauf pendant une exécution)."""

    def __init__(self, ip):
        super().__init__(daemon=True)
        self.ip = ip
        self.verrou = threading.Lock()
        self.dernier = None          # (t, dict etat)
        self.erreur = None
        self.pause = threading.Event()
        self.actif = True

    def run(self):
        while self.actif:
            if not self.pause.is_set():
                try:
                    e = lien_robot.etat(self.ip)
                    with self.verrou:
                        self.dernier = (time.time(), e)
                    self.erreur = None
                except Exception as ex:   # noqa: BLE001
                    self.erreur = str(ex)[:80]
            time.sleep(1.0)

    def lire(self):
        with self.verrou:
            return self.dernier


# ------------------------------------------------------------------------------- vidéo
class Enregistreur:
    """Vidéo MP4 (H.264 avec imageio si disponible, sinon MPEG-4 d'OpenCV), cadence réelle :
    on écrit autant d'images que le temps écoulé l'exige (la vidéo dure ce qu'a duré la scène)."""

    def __init__(self, chemin, taille):
        self.chemin = chemin
        self.t0 = time.time()
        self.n = 0
        self.w = None
        try:
            import imageio.v2 as imageio
            self.w = imageio.get_writer(chemin, fps=FPS_VIDEO, codec="libx264", quality=8,
                                        macro_block_size=16, ffmpeg_params=["-preset", "veryfast"])
            self.type = "imageio"
        except Exception:   # noqa: BLE001
            self.w = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"mp4v"), FPS_VIDEO, taille)
            self.type = "cv2"

    def ecrire(self, img_bgr):
        a_faire = int((time.time() - self.t0) * FPS_VIDEO) - self.n     # si la boucle a pris du retard,
        for _ in range(max(a_faire, 0)):                                 # on répète l'image courante
            if self.type == "imageio":
                self.w.append_data(img_bgr[:, :, ::-1])
            else:
                self.w.write(img_bgr)
            self.n += 1

    def duree(self):
        return time.time() - self.t0

    def fermer(self):
        if self.type == "imageio":
            self.w.close()
        else:
            self.w.release()


VUES_VIDEO = ["webcam", "3/4", "d435"]


def etiquette(img, txt):
    """Titre en bas à gauche (en haut à gauche, il y a déjà les positions détectées)."""
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 36), (12 + 13 * len(txt), h), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def panneau(vue, lignes, largeur=960, hauteur=540):
    out = cv2.resize(vue, (largeur, hauteur)) if vue is not None else np.zeros((hauteur, largeur, 3), np.uint8)
    bandeau = np.zeros((30 * len(lignes) + 12, largeur, 3), np.uint8)
    for i, (txt, col) in enumerate(lignes):
        cv2.putText(bandeau, txt, (10, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 1, cv2.LINE_AA)
    return np.vstack([out, bandeau])


# ------------------------------------------------------------------------------- programme principal
def main():
    ip = arg("--robot")
    vitesse = float(np.clip(arg("--vitesse", 0.1, float), 0.02, 0.3))
    accel = arg("--accel", 1.0, float)
    avec_planche = "--planche" in sys.argv
    politique = arg("--politique", "il")
    if avec_planche:
        import detection_planche as dpl
        from concurrent.futures import ProcessPoolExecutor
        from planif_planche import (OBS_Q, JumeauPlanche, planifier_deplacement, planifier_planche,
                                    prechauffer, rejouer_deplacement)
        # la planification (IK, RRT, choix de l'IL) tourne dans un PROCESSUS à part : dans un thread,
        # elle partagerait le verrou de Python avec la webcam, la D435 et l'affichage (4x plus lente)
        pool = ProcessPoolExecutor(max_workers=1)
        pool.submit(prechauffer)
        from suivi_planche import AGE_MAX_D435, D435Continu, PlancheWebcam, StabilitePlanche
        j = JumeauPlanche()
    else:
        j = Jumeau()
    m, d = j.model, j.data
    detecteur = Detecteur(arg("--image"))
    robot = None
    if ip:
        robot = EtatRobot(ip)
        robot.start()
    stab = Stabilite()
    table_masque = dpl.masque_table(detecteur.cam) if avec_planche else None
    suivi_w = suivi_d = stab_pl = None
    if avec_planche:
        suivi_w = PlancheWebcam(detecteur)
        suivi_w.start()
        stab_pl = StabilitePlanche()
        if "--d435" in sys.argv:
            if robot is None:
                print("[attention] --d435 sans --robot : la D435 a besoin de la pose du bras, ignorée")
            else:
                suivi_d = D435Continu(robot, suivi_w)
                suivi_d.start()

    S = {"mode": "live", "msg": ("Vue en direct. E = simuler l'expert.", BLANC),
         "plan": None, "gen": None, "bilan": None, "exec": None, "exec_res": None,
         "mesure_apres": None, "ecart_fk": None, "video": None, "vue_video": 0,
         "ombre": None, "ligne_robot": "", "planche": None, "planche_masque": None,
         "planche_txt": "Planche : recherche...", "tache": None, "d435_txt": ""}
    os.makedirs(os.path.join(HERE, "essais_img"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "videos"), exist_ok=True)
    rendu = {"r": None}
    cam34 = mujoco.MjvCamera()
    cam34.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam34.lookat[:] = [0.42, 0.0, 0.10]
    cam34.distance, cam34.azimuth, cam34.elevation = 1.25, 150, -28

    # ---------------- caméra du viewer : flèches comme dans twin_scene.py
    cams = [-1] + [m.camera(n).id for n in ("webcam", "d435")]
    cam_etat = {"i": 0, "change": False}

    def touche_viewer(key):
        if key in (262, 263):
            cam_etat["i"] = (cam_etat["i"] + (1 if key == 262 else -1)) % len(cams)
            cam_etat["change"] = True

    from mujoco import viewer
    try:
        ctx = viewer.launch_passive(m, d, key_callback=touche_viewer, show_left_ui=False, show_right_ui=False)
    except TypeError:
        ctx = viewer.launch_passive(m, d, key_callback=touche_viewer)

    def q_robot_frais(max_age=3.0):
        if robot is None:
            return None
        r = robot.lire()
        if r is None or time.time() - r[0] > max_age:
            return None
        return np.array(r[1]["q"])

    # ---------------- cube posé à la main dans la scène (pendant l'ombre du robot réel)
    def poser_cube(pos, R):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        d.qpos[j.cube_q:j.cube_q + 7] = [*pos, *quat]
        d.qvel[j.cube_d:j.cube_d + 6] = 0

    def maj_ombre():
        """Bras virtuel = angles reçus du vrai robot ; le cube suit la pince entre la prise et le lâcher
        (position estimée : la webcam ne voit pas le cube sous la pince)."""
        o = S["ombre"]
        j.poser_bras(o["q"], o["pince"])
        T_tcp = (d.site_xpos[j.tcp].copy(), d.site_xmat[j.tcp].reshape(3, 3).copy())
        if o["attache"] is None and o["pince"] < 0.06:
            p_c, R_c = o["cube"]
            if np.linalg.norm(p_c - T_tcp[0]) < 0.03:
                o["attache"] = (T_tcp[1].T @ (p_c - T_tcp[0]), T_tcp[1].T @ R_c)
        elif o["attache"] is not None and o["pince"] > 0.06:
            p_c, R_c = o["cube"]
            yaw = np.arctan2(R_c[1, 0], R_c[0, 0])
            c, s_ = np.cos(yaw), np.sin(yaw)
            z = red_z(p_c[0], p_c[1]) + TARGET_H + CUBE_HALF
            o["cube"] = (np.array([p_c[0], p_c[1], z]), np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]))
            o["attache"] = None
        if o["attache"] is not None:
            dp, dR = o["attache"]
            o["cube"] = (T_tcp[0] + T_tcp[1] @ dp, T_tcp[1] @ dR)
        poser_cube(*o["cube"])
        mujoco.mj_forward(m, d)

    # ---------------- E : simulation
    def lancer_simulation():
        v = stab.valeur()
        if v is None:
            S["msg"] = ("E refusé : cube ou cible non détecté.", ROUGE)
            return
        if not v[3]:
            S["msg"] = (f"E refusé, scène pas stable : {v[5]}", ROUGE)
            return
        cube, lacet, cible = v[0], v[1], v[2]
        if robot is not None:
            try:
                q0 = np.array(lien_robot.etat(ip)["q"])
            except Exception as ex:   # noqa: BLE001
                S["msg"] = (f"Robot injoignable : {str(ex)[:60]}", ROUGE)
                return
        else:
            q0 = HOME_Q.copy()
        if avec_planche:
            vp = stab_pl.valeur()
            if vp is None or not vp[1]:
                S["msg"] = ("E refusé : planche non vue ou pas stable (" + S["planche_txt"] + ").", ROUGE)
                return
            img_c = image_brute_corrigee()
            S["planche"], S["planche_masque"] = vp[0], (masque_reduit(img_c) if img_c is not None else None)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if img_c is not None:
                cv2.imwrite(os.path.join(HERE, "essais_img", f"{stamp}_E_webcam.png"), img_c)
            rd = suivi_d.lire() if suivi_d else None
            if rd is not None:
                cv2.imwrite(os.path.join(HERE, "essais_img", f"{stamp}_E_d435.png"), rd[2])
            with ctx.lock():
                j.poser_planche(S["planche"])
            lancer_planif_planche(q0, cube, lacet, cible)
            return
        etapes, msg = planifier(j, q0, cube, lacet, cible)
        if etapes is None:
            S["msg"] = (f"Plan impossible : {msg}", ROUGE)
            return
        S["plan"] = {"cube": cube.tolist(), "lacet": lacet, "cible": cible.tolist(),
                     "q_depart": q0.tolist(), "etapes": etapes, "msg_plan": msg}
        S["gen"] = rejouer(j, q0, cube, lacet, cible, etapes)
        S["bilan"] = None
        S["mode"] = "sim"
        S["t_sim"], S["t_horloge"] = 0.0, time.time()
        S["msg"] = (f"Simulation de l'expert... ({msg})", JAUNE)

    # ---------------- P : mesure de la planche (thread : ~3 s webcam, ~5 s de plus avec la D435)
    def image_brute_corrigee():
        with detecteur.verrou:
            im = detecteur.img
        return None if im is None else im[1].copy()

    def masque_reduit(img_c):
        return cv2.resize(dpl.masque_planche(img_c, table_masque), (320, 180), interpolation=cv2.INTER_NEAREST) > 0

    def lancer_observation():
        """O : trajet vers la pose perchée, vérifié dans le jumeau (puis R pour l'exécuter)."""
        if not avec_planche or robot is None:
            S["msg"] = ("O : il faut --planche et --robot.", ROUGE)
            return
        try:
            q0 = np.array(lien_robot.etat(ip)["q"])
        except Exception as ex:   # noqa: BLE001
            S["msg"] = (f"Robot injoignable : {str(ex)[:60]}", ROUGE)
            return
        v = stab.valeur()
        vp = stab_pl.valeur()
        S["mode"] = "planif"
        S["msg"] = ("Trajet vers la pose perchée...", JAUNE)

        fut = pool.submit(planifier_deplacement, q0, OBS_Q, v[0] if v else None, v[1] if v else 0.0,
                          v[2] if v else None, vp[0] if vp else None)

        def fini(f):
            try:
                et, msg = f.result()
                S["tache"] = ("deplacement", (et, q0) if et else None, msg)
            except Exception as ex:   # noqa: BLE001
                S["tache"] = ("deplacement", None, f"erreur : {str(ex)[:70]}")

        fut.add_done_callback(fini)

    def lancer_planif_planche(q0, cube, lacet, cible):
        S["mode"] = "planif"
        S["msg"] = (f"Planification ({politique}) avec la planche... (jusqu'à ~30 s pour un poteau)", JAUNE)
        fut = pool.submit(planifier_planche, q0, cube, lacet, cible, S["planche"], politique)

        def fini(f):
            try:
                et, msg, info = f.result()
                S["tache"] = ("plan", (et, info, q0, cube, lacet, cible), msg)
            except Exception as ex:   # noqa: BLE001
                S["tache"] = ("plan", None, f"erreur de planification : {str(ex)[:70]}")

        fut.add_done_callback(fini)

    def recevoir_tache():
        """Résultats des threads P (mesure) et E (planification), appliqués dans la boucle principale."""
        t, S["tache"] = S["tache"], None
        if t is None:
            return
        quoi, val, msg = t
        if quoi == "deplacement":
            if val is None:
                S["mode"] = "live"
                S["msg"] = (f"O impossible : {msg}", ROUGE)
                return
            et, q0 = val
            S["plan"] = {"type": "deplacement", "q_depart": q0.tolist(), "etapes": et, "msg_plan": msg}
            with ctx.lock():
                S["gen"] = rejouer_deplacement(j, q0, et)
            S["bilan"] = None
            S["mode"] = "sim"
            S["t_sim"], S["t_horloge"] = 0.0, time.time()
            S["msg"] = (f"Simulation du trajet ({msg})", JAUNE)
            return
        if val is None or val[0] is None:
            S["mode"] = "live"
            S["msg"] = (f"Plan impossible : {msg}", ROUGE)
            return
        et, info, q0, cube, lacet, cible = val
        S["plan"] = {"cube": cube.tolist(), "lacet": lacet, "cible": cible.tolist(),
                     "q_depart": q0.tolist(), "etapes": et, "msg_plan": msg, "planche": S["planche"],
                     "politique": politique, "info": info}
        with open(os.path.join(HERE, "dernier_plan_planche.json"), "w") as fj:
            json.dump(vers_json(et, q0, vitesse, {"cube": cube.tolist(), "lacet": lacet, "cible": cible.tolist(),
                                                  "planche": S["planche"], "message": msg}), fj, indent=1)
        with ctx.lock():
            S["gen"] = rejouer(j, q0, cube, lacet, cible, et)
        S["bilan"] = None
        S["mode"] = "sim"
        S["t_sim"], S["t_horloge"] = 0.0, time.time()
        S["msg"] = (f"Simulation... ({msg})", JAUNE)

    # ---------------- R : exécution réelle
    def lancer_execution():
        if S["mode"] != "sim_fini" or not (S["bilan"] and S["bilan"]["ok"]):
            S["msg"] = ("R refusé : il faut d'abord une simulation réussie (touche E).", ROUGE)
            return
        if robot is None:
            S["msg"] = ("R refusé : pas de robot (relancer avec --robot IP).", ROUGE)
            return
        if S["ecart_fk"] is not None and S["ecart_fk"] > ECART_FK_MAX:
            S["msg"] = (f"R refusé : écart sim/réel de {S['ecart_fk']:.0f} mm sur la pince (EE de Desk ?).", ROUGE)
            return
        p = S["plan"]
        deplacement = p.get("type") == "deplacement"
        v = stab.valeur()
        if deplacement:
            pass
        elif v is None:
            S["msg"] = ("R refusé : cube ou cible plus détecté.", ROUGE)
            return
        elif (np.linalg.norm(v[0] - p["cube"]) > TOL_CHANGE_POS or np.linalg.norm(v[2] - p["cible"]) > TOL_CHANGE_POS
                or abs(ecart_lacet(v[1], p["lacet"])) > TOL_CHANGE_LACET):
            S["msg"] = ("R refusé : la scène a bougé depuis la simulation. Refaire E.", ROUGE)
            return
        if avec_planche and not deplacement:
            img_c = image_brute_corrigee()
            if img_c is None or S["planche_masque"] is None:
                S["msg"] = ("R refusé : planche non vérifiable (pas d'image).", ROUGE)
                return
            a, b = masque_reduit(img_c), S["planche_masque"]
            iou = (a & b).sum() / max((a | b).sum(), 1)
            if iou < IOU_PLANCHE_MIN:
                S["msg"] = (f"R refusé : la planche a bougé depuis E (recouvrement {iou:.2f}). Refaire E.", ROUGE)
                return
        q = q_robot_frais()
        if q is None or np.abs(q - np.array(p["q_depart"])).max() > TOL_Q_DEPART:
            S["msg"] = ("R refusé : le robot a bougé depuis la simulation (ou pose non lue). Refaire E.", ROUGE)
            return
        if deplacement:
            traj = vers_json(p["etapes"], p["q_depart"], vitesse, {"deplacement": "pose perchee"})
            c0 = v[0] if v else np.array([0.9, -0.9])
            a = np.radians(v[1]) if v else 0.0
            S["ombre"] = {"q": np.array(p["q_depart"]), "pince": 0.08, "attache": None,
                          "cube": (np.array([*c0, red_z(*c0) + CUBE_HALF]),
                                   np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]))}
            S["ligne_robot"] = ""
            robot.pause.set()
            lancer_thread_execution(traj)
            S["msg"] = (f"Le robot monte en pose perchée (vitesse {vitesse}).", JAUNE)
            return
        traj = vers_json(p["etapes"], p["q_depart"], vitesse,
                         {"cube": p["cube"], "lacet": p["lacet"], "cible": p["cible"],
                          "bilan_sim": S["bilan"]["message"], "planche": p.get("planche"),
                          "politique": p.get("politique")})
        with open(os.path.join(HERE, "derniere_trajectoire.json"), "w") as f:
            json.dump(traj, f, indent=1)
        a = np.radians(p["lacet"])
        S["ombre"] = {"q": np.array(p["q_depart"]), "pince": 0.08, "attache": None,
                      "cube": (np.array([*p["cube"], red_z(*p["cube"]) + CUBE_HALF]),
                               np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]))}
        S["ligne_robot"] = ""
        robot.pause.set()

        lancer_thread_execution(traj)
        S["msg"] = (f"EXECUTION sur le vrai robot (vitesse {vitesse}) - le jumeau suit le robot.", JAUNE)

    def lancer_thread_execution(traj):
        def suivi(msg):
            if "q" in msg:
                S["ombre"]["q"] = np.array(msg["q"])
            elif "pince" in msg:
                S["ombre"]["pince"] = msg["pince"]
            elif "ligne" in msg and not msg["ligne"].startswith("MESURE"):
                S["ligne_robot"] = msg["ligne"]

        def tache():
            try:
                S["exec_res"] = lien_robot.executer(ip, traj, suivi)
            except Exception as ex:   # noqa: BLE001
                S["exec_res"] = {"ok": False, "code": -1, "log": [f"liaison : {ex}"], "mesures": []}
            finally:
                robot.pause.clear()

        S["exec_res"] = None
        S["exec"] = threading.Thread(target=tache, daemon=True)
        S["exec"].start()
        S["mode"] = "execution"

    def enregistrer_essai(erreur_mm, vue):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        f = os.path.join(HERE, "essais_reels.json")
        essais = json.load(open(f)) if os.path.exists(f) else []
        r = S["exec_res"] or {}
        p = S["plan"]
        essais.append({"date": stamp, "cube": p["cube"], "lacet": p["lacet"], "cible": p["cible"],
                       "plan": p["msg_plan"], "vitesse": vitesse, "planche": p.get("planche"),
                       "politique": p.get("politique"), "choix": (p.get("info") or {}).get("choix"),
                       "sim": {k: S["bilan"].get(k) for k in ("message", "erreur_depose_mm")},
                       "robot_ok": r.get("ok"), "robot_code": r.get("code"), "duree_s": r.get("duree_s"),
                       "erreur_reelle_mm": erreur_mm, "log": r.get("log", []), "mesures": r.get("mesures", []),
                       "video": S["video"].chemin if S["video"] else None})
        with open(f, "w") as fp:
            json.dump(essais, fp, indent=1)
        if vue is not None:
            cv2.imwrite(os.path.join(HERE, "essais_img", f"{stamp}.png"), vue)

    def basculer_video():
        if S["video"] is None:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            chemin = os.path.join(HERE, "videos", f"{stamp}_jumeau.mp4")
            S["video"] = Enregistreur(chemin, (2 * VW, VH + BANDEAU_VIDEO))
            print(f"[video] enregistrement -> {chemin} ({S['video'].type})")
        else:
            S["video"].fermer()
            print(f"[video] {S['video'].chemin} : {S['video'].duree():.0f} s")
            S["msg"] = (f"Vidéo enregistrée : videos/{os.path.basename(S['video'].chemin)}", VERT)
            S["video"] = None

    def image_virtuelle():
        if rendu["r"] is None:
            rendu["r"] = mujoco.Renderer(m, VH, VW)
        nom = VUES_VIDEO[S["vue_video"]]
        rendu["r"].update_scene(d, camera=cam34 if nom == "3/4" else nom)
        return rendu["r"].render()[:, :, ::-1].copy()

    print(__doc__.split("Usage")[0])
    with ctx as v:
        v.cam.lookat[:] = [0.55, 0.0, 0.05]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -30
        while v.is_running():
            t0 = time.time()
            snap = detecteur.lire()
            vue = None
            if snap:
                vue = snap[1]
                if snap[3] is not None:
                    stab.ajouter(snap[2], snap[3])
            val = stab.valeur()
            r = robot.lire() if robot else None
            recevoir_tache()
            vp, img_d435 = None, None
            if avec_planche:
                rw = suivi_w.lire()
                rd = suivi_d.lire() if suivi_d else None
                rd_ok = rd if (rd and time.time() - rd[0] < AGE_MAX_D435) else None
                t_new = max(rw[0] if rw else 0.0, rd_ok[0] if rd_ok else 0.0)
                if t_new > 0:
                    stab_pl.ajouter(t_new, rw[1] if rw else None, rd_ok[1] if rd_ok else None)
                vp = stab_pl.valeur()
                S["planche_txt"] = vp[2] if vp else "Planche : non vue"
                if stab_pl.desaccord:
                    suivi_w.forcer_recherche = True          # la webcam refait une recherche complète
                if vp is not None and vp[0] is None:
                    vp = None
                if time.time() - S.get("t_journal", 0.0) > 1.0:          # journal pour le diagnostic
                    S["t_journal"] = time.time()
                    def _pl(x):
                        return (None if not x else
                                f"{x['posture']} ({x['x']:.3f};{x['y']:+.3f}) {np.degrees(x['lacet']):.0f}deg")
                    lig = {"t": datetime.datetime.now().strftime("%H:%M:%S"), "mode": S["mode"],
                           "webcam": _pl(rw[1]) if rw else None,
                           "d435": _pl(rd_ok[1]["planche"]) if rd_ok and rd_ok[1].get("planche") else None,
                           "d435_info": (rd_ok[1].get("info", {}).get("planche_rejet") if rd_ok else "pas de photo"),
                           "fusion": S["planche_txt"],
                           "q": [round(float(v), 3) for v in r[1]["q"]] if r else None,
                           "msg": S["msg"][0]}
                    with open(os.path.join(HERE, "journal_planche.jsonl"), "a", encoding="utf-8") as fj:
                        fj.write(json.dumps(lig, ensure_ascii=False) + "\n")
                if suivi_d:
                    if rd_ok:
                        info = rd_ok[3]
                        rec = (f"recalage tag+table OK (correction {info['decalage_table_mm']:.0f} mm sur la table)"
                               if info.get("ok") else f"sans recalage ({info.get('raison', '?')}) -> O = pose perchée")
                        S["d435_txt"] = f"D435 : photo il y a {time.time() - rd_ok[0]:.0f} s, {rec}"
                        img_d435 = rd_ok[2]
                    else:
                        S["d435_txt"] = ("D435 : " + (suivi_d.erreur or "attend que le bras soit immobile"
                                                      " (O = pose perchée)"))

            with v.lock():
                if cam_etat["change"]:
                    cam_etat["change"] = False
                    c = cams[cam_etat["i"]]
                    if c < 0:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    else:
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        v.cam.fixedcamid = c
                if S["mode"] == "live":
                    q = np.array(r[1]["q"]) if r else (HOME_Q if robot is None else None)
                    if q is not None:
                        larg = r[1].get("largeur", 0.08) if r else 0.08
                        j.poser_bras(q, larg if larg and larg > 0 else 0.08)
                    if val is not None:
                        j.poser_objets(val[0], val[1], val[2])
                    if vp is not None and vp[1]:
                        j.poser_planche(vp[0])
                elif S["mode"] in ("execution", "mesure") and S["ombre"] is not None:
                    maj_ombre()
                elif S["mode"] == "sim":
                    # rejeu en temps réel (x accel) même si la boucle ralentit (rendu vidéo) :
                    # on avance d'autant de tranches de 20 ms que le temps écoulé l'exige
                    S["t_sim"] += (time.time() - S["t_horloge"]) * accel
                    S["t_horloge"] = time.time()
                    n_pas = min(int(S["t_sim"] / 0.02), 100)
                    S["t_sim"] -= n_pas * 0.02
                    for _ in range(n_pas):
                        try:
                            next(S["gen"])
                        except StopIteration as fin:
                            S["bilan"] = fin.value
                            S["mode"] = "sim_fini"
                            ok = S["bilan"]["ok"]
                            suite = "  ->  R = executer sur le robot" if ok and robot else ""
                            S["msg"] = (("SIMULATION OK : " if ok else "SIMULATION ECHEC : ")
                                        + S["bilan"]["message"] + suite, VERT if ok else ROUGE)
                            break
                img_v = image_virtuelle() if S["video"] is not None else None
            v.sync()

            # écart entre le modèle MuJoCo et le vrai robot, à angles égaux
            if r:
                p_sim, _ = j.tcp_fk(np.array(r[1]["q"]))
                p_reel = np.array([r[1]["O_T_EE"][i] for i in (12, 13, 14)])
                S["ecart_fk"] = float(np.linalg.norm(p_sim - p_reel) * 1000)

            # fin d'exécution -> mesure de la position réelle du cube par la webcam
            if S["mode"] == "execution" and S["exec"] is not None and not S["exec"].is_alive():
                res = S["exec_res"]
                if res.get("ok") and S["plan"].get("type") == "deplacement":
                    S["mode"] = "live"
                    S["msg"] = ("Robot en pose perchée : la D435 voit la scène et le tag. E = simuler.", VERT)
                elif res.get("ok"):
                    S["mode"] = "mesure"
                    S["mesure_apres"] = time.time() + 1.5
                    S["msg"] = ("Robot revenu au départ. Mesure du cube par la webcam...", JAUNE)
                else:
                    dern = (res.get("log") or ["?"])[-1]
                    S["mode"] = "fini"
                    S["msg"] = (f"ROBOT : arrêt (code {res.get('code')}) : {dern[:70]}", ROUGE)
                    if S["plan"].get("type") != "deplacement":
                        enregistrer_essai(None, vue)
            if S["mode"] == "mesure" and val is not None and val[3] and val[4] > S["mesure_apres"]:
                err = float(np.linalg.norm(val[0] - np.array(S["plan"]["cible"])) * 1000)
                S["mode"] = "fini"
                with v.lock():                 # le jumeau reprend la position MESURÉE du cube ; la cible,
                    # collée à la table, garde sa position d'avant (posé dessus, le cube en cache une
                    # partie et fausse la détection du disque vert)
                    j.poser_objets(val[0], val[1], np.array(S["plan"]["cible"]))
                S["msg"] = (f"ESSAI REEL TERMINE : cube à {err:.1f} mm de la cible "
                            f"(sim : {S['bilan']['erreur_depose_mm']} mm). L = vue en direct.", VERT)
                enregistrer_essai(round(err, 1), vue)

            # ---------------- textes d'état
            if val is not None:
                c, lac, g, stable = val[:4]
                l1 = (f"Detection : cube ({c[0]:.3f} ; {c[1]:+.3f}) lacet {lac:4.1f} deg   cible ({g[0]:.3f} ; {g[1]:+.3f})"
                      f"   {'STABLE' if stable else 'instable'}", VERT if stable else JAUNE)
            else:
                l1 = (f"Detection : cube ou cible non vu  {detecteur.erreur or ''}", ROUGE)
            if robot is None:
                l2 = ("Robot : non connecte (simulation seule, depart = posture home)", GRIS)
            elif S["mode"] == "execution":
                l2 = (f"Robot : {S['ligne_robot']}", JAUNE)
            elif r is None:
                l2 = (f"Robot : en attente de {ip}...  {robot.erreur or ''}", JAUNE)
            else:
                age = time.time() - r[0]
                fk = S["ecart_fk"]
                l2 = (f"Robot : pose lue il y a {age:.0f} s   ecart pince sim/reel {fk:.1f} mm",
                      VERT if fk is not None and fk <= ECART_FK_MAX and age < 3 else JAUNE)
            rec = (f"REC {S['video'].duree():4.0f} s  (V = arreter, C = vue du jumeau : {VUES_VIDEO[S['vue_video']]})"
                   if S["video"] else "V = enregistrer une video")
            aide = (("O = pose perchee   " if avec_planche and robot else "") + "E = simuler   R = executer   L = vue en direct   "
                    + rec + "   Q = quitter", ROUGE if S["video"] else GRIS)
            lignes_pl = []
            if avec_planche:
                ok_pl = vp is not None and vp[1]
                lignes_pl.append((S["planche_txt"], VERT if ok_pl else JAUNE))
                if suivi_d:
                    lignes_pl.append((S["d435_txt"], VERT if img_d435 is not None else JAUNE))
                if vue is not None and img_d435 is not None:      # vignette D435 en haut à droite
                    vue = vue.copy()
                    vig = cv2.resize(img_d435, (320, 180))
                    vue[0:180, vue.shape[1] - 320:] = vig
                    cv2.rectangle(vue, (vue.shape[1] - 320, 0), (vue.shape[1] - 1, 179), (255, 255, 255), 1)

            # ---------------- vidéo : réel | jumeau, avec le bandeau d'état
            if S["video"] is not None and vue is not None:
                gauche = etiquette(cv2.resize(vue, (VW, VH)), "REEL - webcam")
                droite = etiquette(img_v, f"JUMEAU - MuJoCo (vue {VUES_VIDEO[S['vue_video']]})")
                bas = np.zeros((BANDEAU_VIDEO, 2 * VW, 3), np.uint8)
                for i, (txt, col) in enumerate([l1, S["msg"], l2]):
                    cv2.putText(bas, txt, (12, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)
                compo = np.vstack([np.hstack([gauche, droite]), bas])
                S["video"].ecrire(compo)
                ecran = panneau(cv2.resize(np.hstack([gauche, droite]), (1440, 405)),
                                [l1] + lignes_pl + [l2, S["msg"], aide], largeur=1440, hauteur=405)
            else:
                ecran = panneau(vue, [l1] + lignes_pl + [l2, S["msg"], aide])
            cv2.imshow("Jumeau", ecran)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("e") and S["mode"] in ("live", "sim_fini", "fini"):
                lancer_simulation()
            elif k == ord("o") and S["mode"] in ("live", "sim_fini", "fini"):
                lancer_observation()
            elif k == ord("r"):
                lancer_execution()
            elif k == ord("l") and S["mode"] in ("sim_fini", "fini", "sim"):
                S["mode"] = "live"
                S["msg"] = ("Vue en direct. E = simuler l'expert.", BLANC)
            elif k == ord("v"):
                basculer_video()
            elif k == ord("c"):
                S["vue_video"] = (S["vue_video"] + 1) % len(VUES_VIDEO)
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))     # ~50 images/s, rejeu sim en temps réel x accel

    if S["video"] is not None:
        basculer_video()
    detecteur.actif = False
    for t in (suivi_w, suivi_d):
        if t is not None:
            t.actif = False
    if avec_planche:
        pool.shutdown(wait=False, cancel_futures=True)
    if robot:
        robot.actif = False
    if rendu["r"] is not None:
        rendu["r"].close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
