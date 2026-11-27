"""Suivi EN CONTINU de la planche pour le jumeau (26/11) : webcam + D435 au poignet.

  PlancheWebcam  (thread) : ajustement de la boîte 3D sur la dernière image webcam. Recherche
                  complète (~3 s) au début ou quand on la perd, puis suivi local autour de
                  l'estimation précédente (~0,3 s) -> environ 2 mises à jour par seconde.
                  27/11 : quand on déplace la planche (désaccord webcam / D435), la recherche
                  repart de la planche vue par la D435 (recherche locale, rapide) au lieu d'une
                  recherche complète (10-15 s vus dans le journal du 26/11, threads + MuJoCo).
  D435Continu    (thread) : flux D435 ouvert en permanence ; dès que le bras est IMMOBILE (deux
                  lectures successives de la pose à moins de 1 mm), photo (médiane de 5 images
                  de profondeur), RECALAGE sur le tag 0 et la table (recalage_d435.py), puis
                  détection planche / cube / cible par la profondeur (~1 par seconde).
                  L'amas D435 retenu est celui qui est près de la planche vue par la webcam
                  (le boîtier de commande Franka, sur la table, ressemblait à une planche).
  27/11 (2) : mesuré dans le jumeau : ajustement local 1,1 s, recherche complète 17-18 s. Donc :
    - la D435 (3D, recalée) devient la référence : quand elle voit toute la planche, elle suffit,
      même si la webcam n'est pas d'accord (il faut 3 photos D435 distinctes et cohérentes) ;
    - si aucun amas D435 n'est près de la planche webcam, la D435 refait son choix sans la webcam
      (forme de la planche) : une webcam fausse ne peut plus cacher la vraie planche ;
    - la webcam seule ne peut pas donner STABLE dans les 10 s qui suivent un désaccord.
  27/11 (3) : l'ajustement webcam tourne dans un PROCESSUS à part. Mesuré : recherche complète
    ~2 s seule (cloud) / ~3 s seule (Windows), mais 17-18 s dans le jumeau (threads Python qui se
    partagent le GIL avec le viewer MuJoCo, la vidéo et la D435). Un processus a son propre cœur.
  StabilitePlanche : fusion des deux sources (detection_planche.scene_depuis : D435 3/4 + webcam
                  1/4, lacet et posture de la D435 quand elle voit toute la planche) + filtre :
                  la planche est « stable » si les 3 dernières estimations ont la même posture et
                  sont à moins de 1,5 cm / 5° les unes des autres (lacet non compté pour un poteau).
"""
import collections
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "perception"))
import detection_planche as dpl  # noqa: E402
import recalage_d435 as rc  # noqa: E402

N_STABLE_PL = 3
TOL_PL_POS, TOL_PL_LACET = 0.015, 5.0
AGE_MAX_D435 = 8.0          # s : au-delà, la mesure D435 n'est plus utilisée
ECART_MAX_MM = 50.0         # au-delà, webcam et D435 sont en désaccord
DELAI_WEBCAM_SEULE = 10.0   # s : après un désaccord, la webcam seule ne suffit pas pour STABLE


# ------------------------------------------------ ajustement webcam dans un processus séparé
_CAM_W, _TABLE_W = None, None


def _init_processus():
    """Exécuté une fois dans le processus de calcul : caméra idéale + masque de la table."""
    global _CAM_W, _TABLE_W
    import detection as det

    class _WebcamRapide(det.Webcam):          # comme dans jumeau_live : inverse de la pose en cache
        def __init__(self):
            super().__init__()
            self.T_cb = np.linalg.inv(self.T_bc)

        def point_vers_pixel(self, p):
            q = self.T_cb @ np.r_[p, 1.0]
            return self.Ki[0, 0] * q[0] / q[2] + self.Ki[0, 2], self.Ki[1, 1] * q[1] / q[2] + self.Ki[1, 2]

    _CAM_W = _WebcamRapide()
    _TABLE_W = dpl.masque_table(_CAM_W)


def _ajuster_processus(img_c, depart):
    return dpl.planche_webcam(img_c, _CAM_W, _TABLE_W, depart=depart)


def ecart_lacet_pi(a, b):
    d = (a - b) % np.pi
    return min(d, np.pi - d)


class PlancheWebcam(threading.Thread):
    def __init__(self, detecteur):
        super().__init__(daemon=True)
        self.det = detecteur
        self.table = dpl.masque_table(detecteur.cam)
        self.verrou = threading.Lock()
        self.res = None                  # (t, planche dict ou None, contour)
        self.actif = True
        self.pause = threading.Event()
        self.forcer_recherche = False    # mis à True si la D435 voit la planche ailleurs
        self.d435 = None                 # D435Continu (renseigné par lui) : sert d'indice de départ
        self.duree = None                # durée du dernier ajustement (s)
        self.pool = None                 # processus de calcul (créé dans run)

    def indice_d435(self):
        """Planche vue par la D435 il y a moins de 3 s, au format 'depart' de ajuster_planche."""
        rd = self.d435.lire() if self.d435 else None
        if not rd or time.time() - rd[0] > 3.0 or not rd[1].get("planche"):
            return None
        p = rd[1]["planche"]
        return {"posture": p["posture"], "x": p["x"], "y": p["y"], "lacet": p["lacet"]}

    def _ajuster(self, img_c, depart):
        """Dans le processus séparé si possible, sinon ici (comme avant)."""
        if self.pool is not None:
            try:
                return self.pool.submit(_ajuster_processus, img_c, depart).result(timeout=60)
            except Exception as ex:   # noqa: BLE001
                print(f"[webcam] processus séparé indisponible ({str(ex)[:60]}) : calcul dans le jumeau")
                self.pool = None
        return dpl.planche_webcam(img_c, self.det.cam, self.table, depart=depart)

    def run(self):
        prec, t_fait = None, 0.0
        try:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                               initializer=_init_processus)
            self.pool.submit(int, 0).result(timeout=120)          # démarrage (imports) une fois
            print("[webcam] ajustement de la planche dans un processus séparé")
        except Exception as ex:   # noqa: BLE001
            print(f"[webcam] pas de processus séparé ({str(ex)[:60]}) : calcul dans le jumeau")
            self.pool = None
        while self.actif:
            if self.pause.is_set():
                time.sleep(0.1)
                continue
            with self.det.verrou:
                im = self.det.img
            if im is None or im[0] <= t_fait:
                time.sleep(0.05)
                continue
            t_fait = im[0]
            if self.forcer_recherche:          # planche déplacée : on repart de ce que voit la D435
                self.forcer_recherche = False     # (None -> recherche complète, comme avant)
                prec = self.indice_d435()
            t0 = time.time()
            try:
                pl, ct = self._ajuster(im[1], prec)
            except Exception:   # noqa: BLE001
                pl, ct = None, None
            self.duree = time.time() - t0
            if self.duree > 2.5:
                print(f"[webcam] ajustement planche lent : {self.duree:.1f} s "
                      f"({'indice D435' if prec else 'sans indice'})")
            prec = pl if (pl is not None and pl["iou"] > 0.8) else None
            with self.verrou:
                self.res = (im[0], pl, ct)
            time.sleep(0.2)

    def lire(self):
        with self.verrou:
            return self.res


class D435Continu(threading.Thread):
    """Flux D435 permanent + détection dès que le bras est immobile. robot = EtatRobot du jumeau."""

    def __init__(self, robot, suivi_webcam=None):
        super().__init__(daemon=True)
        self.robot = robot
        self.suivi_webcam = suivi_webcam     # planche vue par la webcam : choisit l'amas D435
        if suivi_webcam is not None:
            suivi_webcam.d435 = self         # et la webcam repart de la D435 quand on déplace la planche
        self.verrou = threading.Lock()
        self.res = None                  # (t, résultats detecter_d435, image annotée, info recalage, T_bc)
        self.erreur = None
        self.actif = True
        self.pause = threading.Event()
        self.T_ec = dpl.T_EE_cam()
        self.recalage = None             # dernière main-oeil recalée (T_EE_cam), réutilisée sans tag

    def run(self):
        try:
            import pyrealsense2 as rs
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            prof = pipe.start(cfg)
            echelle = prof.get_device().first_depth_sensor().get_depth_scale()
            for capteur in prof.get_device().query_sensors():
                if capteur.supports(rs.option.enable_auto_white_balance):
                    capteur.set_option(rs.option.enable_auto_white_balance, 1)
            align = rs.align(rs.stream.color)
        except Exception as ex:   # noqa: BLE001
            self.erreur = f"D435 : {str(ex)[:60]}"
            return
        pile = collections.deque(maxlen=5)
        dernier_q, t_dernier = None, 0.0
        while self.actif:
            try:
                fr = align.process(pipe.wait_for_frames(2000))
            except Exception as ex:   # noqa: BLE001
                self.erreur = f"D435 : {str(ex)[:60]}"
                continue
            pile.append(np.asanyarray(fr.get_depth_frame().get_data()).astype(np.float32))
            if self.pause.is_set() or time.time() - t_dernier < 1.0 or len(pile) < 5:
                continue
            r = self.robot.lire() if self.robot else None
            if r is None or time.time() - r[0] > 2.5:
                continue
            q = np.array(r[1]["q"])
            if dernier_q is None or r[0] != dernier_q[0]:          # nouvelle lecture de la pose
                immobile = dernier_q is not None and np.abs(q - dernier_q[1]).max() < 0.002
                dernier_q = (r[0], q, immobile)
            if not dernier_q[2]:
                continue                                         # bras en mouvement (ou 1re lecture)
            t_dernier = time.time()
            couleur = np.asanyarray(fr.get_color_frame().get_data()).copy()
            intr = fr.get_color_frame().profile.as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
            P = np.array(pile)
            P[P == 0] = np.nan
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    prof_m = np.nan_to_num(np.nanmedian(P, axis=0)) * echelle
            T_be = np.array(r[1]["O_T_EE"]).reshape(4, 4).T
            T_chaine = T_be @ (self.recalage if self.recalage is not None else self.T_ec)
            try:
                T_bc, info = rc.recaler(prof_m, couleur, K, T_chaine)
                if info["ok"]:
                    self.recalage = np.linalg.inv(T_be) @ T_bc
                    rc.sauver(T_be, T_bc, info)
                ref = None
                rw = self.suivi_webcam.lire() if self.suivi_webcam else None
                if rw and rw[1] is not None and time.time() - rw[0] < 5.0:
                    ref = rw[1]
                d435 = dpl.detecter_d435(prof_m, couleur, K, T_bc, ref=ref)
                if ref is not None and d435.get("planche") is None and "15 cm" in str(
                        d435.get("info", {}).get("planche_rejet", "")):
                    # la webcam se trompe peut-être (planche déplacée) : choix par la forme seule
                    d435 = dpl.detecter_d435(prof_m, couleur, K, T_bc, ref=None)
                    if d435.get("planche") is not None:
                        d435["planche"]["choisi"] = "forme seule (loin de la planche webcam)"
                d435["t"] = t_dernier
                img = dpl.annoter_d435(couleur, d435, K, T_bc)
                self.erreur = None
            except Exception as ex:   # noqa: BLE001
                self.erreur = f"D435 : {str(ex)[:60]}"
                continue
            with self.verrou:
                self.res = (t_dernier, d435, img, info, T_bc)
        pipe.stop()

    def lire(self):
        with self.verrou:
            return self.res


class StabilitePlanche:
    def __init__(self):
        self.h = collections.deque(maxlen=N_STABLE_PL)
        self.t_dernier = 0.0
        self.t_d435 = None               # dernière photo D435 ajoutée seule (une photo = une entrée)
        self.ecarts = {}
        self.desaccord = False
        self.t_desaccord = 0.0

    @staticmethod
    def d435_fiable(pd):
        return (pd is not None and not pd.get("attention")
                and (pd["posture"] != "debout" or pd.get("hauteur_mesuree_cm", 0) > 45))

    def ajouter(self, t, webcam_pl, d435_res):
        if t <= self.t_dernier:
            return
        self.t_dernier = t
        w = {"planche": webcam_pl} if webcam_pl is not None else None
        sc, self.ecarts = dpl.scene_depuis(w, d435_res)
        self.desaccord = (self.ecarts.get("planche_xy_mm", 0) > ECART_MAX_MM
                          or self.ecarts.get("posture", "identique") != "identique")
        if self.desaccord:
            self.t_desaccord = time.time()
        pd = d435_res.get("planche") if d435_res else None
        if self.d435_fiable(pd) and self.desaccord:
            # webcam en retard ou trompée : la D435 seule, une entrée par photo
            t_d = d435_res.get("t")
            if t_d is not None and t_d == self.t_d435:
                return
            self.t_d435 = t_d
            self.h.append({"posture": pd["posture"], "x": pd["x"], "y": pd["y"], "lacet": pd["lacet"],
                           "source": "D435"})
            return
        if "posture" not in sc or self.desaccord:
            # D435 incomplète et pas d'accord avec la webcam : on attend
            self.h.clear()
            return
        source = "fusion" if (pd is not None and webcam_pl is not None) else ("D435" if pd else "webcam")
        self.h.append({"posture": sc["posture"], "x": sc["planche_xy"][0], "y": sc["planche_xy"][1],
                       "lacet": sc["planche_lacet"], "source": source})

    def valeur(self):
        """(planche filtrée, stable, texte) ou None."""
        if not self.h:
            if self.desaccord:
                e = self.ecarts
                return None, False, (f"Planche : webcam et D435 en DESACCORD ({e.get('planche_xy_mm', 0):.0f} mm, "
                                     f"posture {e.get('posture', '?')}) : attendre / sortir la main du champ")
            return None
        postures = [e["posture"] for e in self.h]
        po = max(set(postures), key=postures.count)
        E = [e for e in self.h if e["posture"] == po]
        xy = np.median([[e["x"], e["y"]] for e in E], axis=0)
        lac = np.array([e["lacet"] for e in E]) * 2              # moyenne circulaire modulo pi
        lacet = float(np.arctan2(np.sin(lac).mean(), np.cos(lac).mean()) / 2 % np.pi)
        dp = max(np.hypot(e["x"] - xy[0], e["y"] - xy[1]) for e in E)
        dl = max(np.degrees(ecart_lacet_pi(e["lacet"], lacet)) for e in E)
        stable = (len(E) == N_STABLE_PL and len(self.h) == N_STABLE_PL and dp < TOL_PL_POS
                  and (po == "debout" or dl < TOL_PL_LACET))
        sources = {e["source"] for e in self.h}
        attente = sources == {"webcam"} and time.time() - self.t_desaccord < DELAI_WEBCAM_SEULE
        if attente:
            stable = False
        pl = {"posture": po, "x": float(xy[0]), "y": float(xy[1]), "lacet": lacet}
        txt = f"Planche : {po} ({xy[0]:.3f} ; {xy[1]:+.3f}) lacet {np.degrees(lacet):.0f} deg"
        if self.h[-1]["source"] == "D435" and self.desaccord:
            txt += f" [D435 seule, webcam a {self.ecarts.get('planche_xy_mm', 0):.0f} mm]"
        elif self.ecarts.get("planche_xy_mm") is not None:
            txt += f" (webcam/D435 {self.ecarts['planche_xy_mm']:.0f} mm)"
        elif sources == {"webcam"}:
            txt += " [webcam seule" + (", attente D435]" if attente else "]")
        txt += "  STABLE" if stable else "  instable"
        return pl, stable, txt
