"""Erreur de la détection webcam, mesurée contre la vérité terrain du robot (pointage au crayon).

Entrées (dans perception/) :
    detection_essais.json : détections enregistrées avec une étiquette (detection.py, ESPACE + e1, e2...)
    verite_detection.txt  : sortie de lire_pose sur l'Ubuntu avec les points
                            cube_e1, cible_e1, cube_e2, cible_e2, ... et blanche1 (longueur du crayon)
                            - cube_eN  : pointe au CENTRE DU DESSUS du cube
                            - cible_eN : pointe au CENTRE de la cible
Usage :  python perception/erreur_detection.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ess = json.load(open(os.path.join(HERE, "detection_essais.json")))
    P = {}
    for line in open(os.path.join(HERE, "verite_detection.txt")):
        a = line.split()
        if len(a) >= 13:
            P[a[0]] = (np.array(a[1:4], float), np.array(a[4:13], float).reshape(3, 3))
    blanches = [k for k in P if k.startswith("blanche")]
    if not blanches:
        raise SystemExit("Il faut au moins un point 'blanche1' (table blanche) pour la longueur du crayon.")
    d = np.mean([-P[k][0][2] / P[k][1][2, 2] for k in blanches])
    pointe = {k: P[k][0] + d * P[k][1][:, 2] for k in P}
    print(f"Longueur du crayon : {d * 1000:.1f} mm\n")
    erreurs = {"cube": [], "cible": []}
    for e, det in sorted(ess.items()):
        for obj in ("cube", "cible"):
            k = f"{obj}_{e}"
            if k not in pointe:
                continue
            if det.get(obj) is None:
                print(f"  {e} {obj:5s} : non détecté")
                continue
            vrai = pointe[k][:2]
            mesure = np.array(det[obj]["xyz"][:2])
            err = mesure - vrai
            erreurs[obj].append(err)
            print(f"  {e} {obj:5s} : vrai ({vrai[0]:.4f} ; {vrai[1]:.4f})  détecté ({mesure[0]:.4f} ; {mesure[1]:.4f})"
                  f"  erreur {np.linalg.norm(err) * 1000:5.1f} mm  (dx {err[0] * 1000:+.1f}, dy {err[1] * 1000:+.1f})")
    print()
    for obj, L in erreurs.items():
        if L:
            L = np.array(L) * 1000
            n = np.linalg.norm(L, axis=1)
            print(f"{obj:5s} : erreur moyenne {n.mean():.1f} mm, max {n.max():.1f} mm, "
                  f"biais moyen (dx {L[:, 0].mean():+.1f}, dy {L[:, 1].mean():+.1f}) mm sur {len(L)} essais")


if __name__ == "__main__":
    main()
