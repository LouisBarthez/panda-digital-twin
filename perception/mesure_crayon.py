"""Série de mesures « webcam contre crayon » guidée pas à pas, depuis le Windows SEUL (26/11).

Tout se fait dans CETTE console : le script prend lui-même les photos webcam et lit la position du
crayon sur le robot par le réseau (serveur_robot.py de l'Ubuntu, comme le jumeau). Rien à taper sur
l'Ubuntu pendant la série.

Avant : - Ubuntu : serveur_robot.py lancé (rien d'autre qui parle au robot : pas de lire_pose) ;
        - robot en lumière BLEUE : le script commence par ouvrir la pince, te laisse glisser le
          crayon, puis le serre (27/11) ; ensuite tu passes en guidage pour pointer.

Usage (PowerShell, venv activé, à la racine de pfe_mujoco) :
    python perception/mesure_crayon.py --ip 10.254.79.65
    python perception/mesure_crayon.py --ip 10.254.79.65 --essais 10      (nombre d'essais, 10 par défaut)
On peut arrêter (q) et relancer : la série reprend au premier essai incomplet.

Déroulé :
  0. pointe du crayon posée sur la table BLANCHE (longueur du crayon) ;
  pour chaque essai dN :
  1. tu places cube et cible, bras hors du champ -> Entrée -> PHOTO webcam (le script vérifie
     qu'il voit le cube et la cible, sinon il redemande) ;
  2. pointe sur un coin du DESSUS du cube -> Entrée ;
  3. pointe sur le coin OPPOSÉ en diagonale -> Entrée ;
  4. pointe au centre de la cible -> Entrée ;
  à la fin : calcul (evaluer_crayon.py) et nouvelle correction dans correction_detection.json.
Fichiers : perception/verite_crayon_d.txt (points), perception/essais_img/dN.png (photos brutes).
"""
import argparse
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "jumeau"))
sys.argv_orig = list(sys.argv)
VERITE = os.path.join(HERE, "verite_crayon_d.txt")
IMG = os.path.join(HERE, "essais_img")

PLAN = [  # (cube, angle du cube, cible) : suggestions pour couvrir la zone de travail
    ("près du robot, à gauche", "0°", "loin, à droite"),
    ("près du robot, au centre", "20°", "loin, à gauche"),
    ("près du robot, à droite", "45°", "milieu, à gauche"),
    ("milieu, à gauche", "60°", "près, à droite"),
    ("milieu, au centre", "70°", "loin, au centre"),
    ("milieu, à droite", "10°", "près, à gauche"),
    ("loin, à gauche", "50°", "milieu, à droite"),
    ("loin, à droite", "85°", "près, au centre"),
    ("AU SOLEIL", "au choix", "à l'ombre"),
    ("à l'OMBRE", "au choix", "au soleil"),
]


class Camera:
    """Webcam lue en continu dans un fil à part : la photo prise est toujours la plus récente."""

    def __init__(self):
        from detection import ouvrir_webcam
        self.cap = ouvrir_webcam()
        self.img = None
        self.ok = True
        threading.Thread(target=self._boucle, daemon=True).start()
        t0 = time.time()
        while self.img is None and time.time() - t0 < 10:
            time.sleep(0.1)
        if self.img is None:
            raise SystemExit("[ERREUR] pas d'image webcam (webcam_index.py ? appli Caméra ouverte ?)")

    def _boucle(self):
        while self.ok:
            ok, im = self.cap.read()
            if ok:
                self.img = im

    def photo(self):
        time.sleep(0.5)
        return self.img.copy()


def lire_points():
    P = {}
    if os.path.exists(VERITE):
        for line in open(VERITE):
            a = line.split()
            if len(a) >= 13:
                P[a[0]] = line
    return P


def mesurer(ip, etiquette, n=3):
    """Moyenne de n lectures de O_T_EE (robot immobile, ~1 s chacune) -> ligne « etiquette x y z r11..r33 »."""
    from lien_robot import etat
    Ts = []
    print("    mesure... (ne bouge pas)", flush=True)
    for _ in range(n):
        Ts.append(np.array(etat(ip)["O_T_EE"]).reshape(4, 4).T)       # libfranka : colonne-major
        time.sleep(0.05)
    Ts = np.array(Ts)
    if np.ptp(Ts[:, :3, 3], axis=0).max() > 0.001:
        print("    [!] le bras a bougé pendant la mesure : on recommence ce point")
        return None
    T = Ts.mean(0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if abs(R[2, 2]) < 0.995:
        print(f"    [attention] crayon penché de {np.degrees(np.arccos(abs(R[2, 2]))):.1f}° "
              "(corrigé par le calcul, mais vise plus vertical si tu peux)")
    p = T[:3, 3]
    return f"{etiquette} " + " ".join(f"{v:.6f}" for v in np.r_[p, R.flatten()])


def demander(texte):
    r = input(f"\n>>> {texte}\n    Entrée = mesurer, r = refaire le point précédent, q = arrêter : ").strip().lower()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="IP de l'Ubuntu (serveur_robot.py)")
    ap.add_argument("--essais", type=int, default=10)
    args = ap.parse_args()
    from lien_robot import etat
    try:
        e = etat(args.ip)
    except Exception as ex:
        raise SystemExit(f"[ERREUR] robot injoignable par {args.ip} ({ex}). serveur_robot.py tourne ? "
                         "lire_pose est bien fermé ?")
    print(f"Robot OK. Pince : {e['largeur'] * 1000:.1f} mm d'ouverture.")
    from lien_robot import serrer_crayon
    if not serrer_crayon(args.ip):
        raise SystemExit("Arrêt : crayon pas serré. Corrige puis relance la même commande.")
    input("\n>>> Passe le robot en GUIDAGE pour déplacer le crayon à la main, puis Entrée : ")
    os.makedirs(IMG, exist_ok=True)
    sys.argv = [sys.argv[0], "--brut"]
    import detection as D
    D.METHODE = "modele"
    D.CORRECTION = {"cube": None, "cible": None}
    wcam = D.Webcam()
    seuils = D.charger_seuils()
    cam = Camera()

    # suite des étapes : (étiquette, consigne, type)
    etapes = [("blanche1", "Pose la pointe du crayon sur la TABLE BLANCHE (près de la base), crayon vertical.", "point")]
    for i in range(args.essais):
        k = f"d{i + 1}"
        cu, an, ci = PLAN[i % len(PLAN)]
        etapes += [
            (k, f"ESSAI {k} : place le CUBE ({cu}, angle ~{an}) et la CIBLE ({ci}).\n"
                f"    Sors le bras du champ de la webcam. Ne touche plus ni cube ni cible. Entrée = PHOTO.", "photo"),
            (f"cube_{k}_a", f"{k} : pointe sur un COIN du DESSUS du cube.", "point"),
            (f"cube_{k}_b", f"{k} : pointe sur le coin OPPOSÉ en diagonale (même dessus).", "point"),
            (f"cible_{k}", f"{k} : pointe au CENTRE de la cible.", "point"),
        ]

    P = lire_points()
    i = 0
    while i < len(etapes):
        et, consigne, typ = etapes[i]
        fait = (et in P) if typ == "point" else os.path.exists(os.path.join(IMG, f"{et}.png"))
        if fait and not getattr(main, "refaire", False):
            i += 1
            continue
        main.refaire = False
        print(f"\n[{i + 1}/{len(etapes)}]", end="")
        r = demander(consigne)
        if r == "q":
            print("Arrêt. Relance la même commande pour reprendre là où tu en es.")
            break
        if r == "r" and i > 0:
            i -= 1
            main.refaire = True
            continue
        if typ == "photo":
            img = cam.photo()
            res, _ = D.detecter(wcam.corriger(img), wcam, seuils)
            manque = [o for o in ("cube", "cible") if res.get(o) is None]
            ann = D.annoter(wcam.corriger(img), res, wcam)
            cv2.imshow("Photo de l'essai", cv2.resize(ann, (960, 540)))
            cv2.waitKey(1200)
            if manque:
                print(f"    [!] la webcam ne voit pas : {', '.join(manque)} (bras dans le champ ? objet hors table ?)"
                      " -> on refait la photo")
                continue
            cv2.imwrite(os.path.join(IMG, f"{et}.png"), img)
            print(f"    photo {et} OK (cube et cible vus)")
            # nouvelle photo = les points de cet essai sont à refaire
            P = {k: v for k, v in P.items() if not k.endswith(f"_{et}") and f"_{et}_" not in k}
        else:
            ligne = mesurer(args.ip, et)
            if ligne is None:
                continue
            P[et] = ligne + "\n"
            v = [float(x) for x in ligne.split()[1:4]]
            print(f"    {et} : x = {v[0]:.4f}  y = {v[1]:.4f}  z = {v[2]:.4f} m (bride)")
        with open(VERITE, "w") as f:
            f.writelines(P[k] for k in sorted(P))
        i += 1

    cam.ok = False
    cv2.destroyAllWindows()
    n_ok = sum(1 for k in P if k.startswith("cible_d"))
    print(f"\n{len(P)} points dans verite_crayon_d.txt ({n_ok} essais complets).")
    if n_ok >= 4:
        print("Calcul de la précision et de la nouvelle correction :\n")
        subprocess.run([sys.executable, os.path.join(HERE, "evaluer_crayon.py"), "--verite", "verite_crayon_d.txt"])


if __name__ == "__main__":
    main()
