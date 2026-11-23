"""Vérifie le lacet (orientation) du cube détecté par la webcam, contre la pince du robot.

Pour chaque essai lN : le cube est détecté (detection.py, ESPACE, étiquette lN), puis la pince
OUVERTE est descendue autour du cube, doigts bien à plat contre deux faces opposées, et sa pose est
lue (lire_pose --libre --sans-pince, étiquette pince_lN). Les doigts se ferment selon l'axe y de
l'EE : les faces du cube sont donc perpendiculaires à cet axe -> lacet du cube = lacet de l'EE (mod 90°).

Entrées (dans perception/) : detection_essais.json, verite_lacet.txt
Usage :  python perception/verifier_lacet.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ecart_mod90(a, b):
    """Écart entre deux angles définis à 90° près (symétrie du cube), dans [-45 ; 45]."""
    return (a - b + 45.0) % 90.0 - 45.0


def main():
    ess = json.load(open(os.path.join(HERE, "detection_essais.json")))
    poses = {}
    for line in open(os.path.join(HERE, "verite_lacet.txt")):
        a = line.split()
        if len(a) >= 13:
            poses[a[0]] = np.array(a[4:13], float).reshape(3, 3)
    ecarts = []
    for e in sorted(ess):
        k = f"pince_{e}"
        if k not in poses:
            continue
        cube = ess[e].get("cube")
        if not cube or "lacet_deg" not in cube:
            print(f"  {e} : cube non détecté")
            continue
        R = poses[k]
        lacet_pince = np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 90.0
        lacet_det = cube["lacet_deg"] % 90.0
        d = ecart_mod90(lacet_det, lacet_pince)
        ecarts.append(d)
        print(f"  {e} : pince {lacet_pince:5.1f}°   détecté {lacet_det:5.1f}°   écart {d:+5.1f}°")
    if not ecarts:
        raise SystemExit("Aucun essai apparié (étiquettes lN côté Windows et pince_lN côté Ubuntu).")
    ecarts = np.array(ecarts)
    print(f"\nLacet : écart moyen {np.abs(ecarts).mean():.1f}°, max {np.abs(ecarts).max():.1f}°, "
          f"biais {ecarts.mean():+.1f}° sur {len(ecarts)} essais")


if __name__ == "__main__":
    main()
