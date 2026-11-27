"""Compare les deux méthodes de détection du cube sur des essais RÉELS (images enregistrées),
contre la vérité donnée par la pince du robot, et apprend la correction fine de la meilleure.

Essais : detection.py en direct, ESPACE + étiquette c1, c2... -> image brute dans essais_img/c1.png.
Vérité : lire_pose --libre --sans-pince, pince OUVERTE descendue autour du cube, doigts à plat
contre deux faces opposées, bien centrée -> étiquette pince_c1, pince_c2...
         - centre du cube (x, y) = position de l'EE (point entre les doigts) ;
         - lacet du cube = lacet de l'EE modulo 90° (les doigts se ferment selon y de l'EE).
Fichier de vérité : perception/verite_cube.txt

Usage :  python perception/evaluer_detection.py
         python perception/evaluer_detection.py --serie d --verite verite_cube_d.txt
             (n'utilise que les essais dont l'étiquette commence par « d » : indispensable après un
              recalage de la webcam, les anciennes images ne correspondent plus à la nouvelle pose)
Affiche, pour la méthode « centroide » (1re version) et « modele » (ajustement d'un cube 3D) :
erreur de position et de lacet bruts, puis erreur de position après correction affine en validation
croisée (un essai laissé de côté). Enregistre la correction du cube pour la meilleure méthode.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def ecart_mod90(a, b):
    return (a - b + 45.0) % 90.0 - 45.0


def _arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


def charger_verite():
    V = {}
    for line in open(os.path.join(HERE, _arg("--verite", "verite_cube.txt"))):
        a = line.split()
        if len(a) >= 13 and a[0].startswith("pince_"):
            p = np.array(a[1:4], float)
            R = np.array(a[4:13], float).reshape(3, 3)
            V[a[0][len("pince_"):]] = (p[:2], np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 90.0)
    return V


def main():
    serie = _arg("--serie", "")
    V = {k: v for k, v in charger_verite().items() if k.startswith(serie)}
    labels = [k for k in sorted(V) if os.path.exists(os.path.join(HERE, "essais_img", f"{k}.png"))]
    manquants = [k for k in V if k not in labels]
    if manquants:
        print(f"(pas d'image pour : {manquants})")
    if len(labels) < 4:
        raise SystemExit("Il faut au moins 4 essais avec image (essais_img/<etiquette>.png) et vérité pince_<etiquette>.")

    sys.argv = [sys.argv[0], "--brut"]
    import detection as D
    cam = D.Webcam()
    seuils = D.charger_seuils()
    res = {}
    for meth in ("centroide", "modele"):
        D.METHODE = meth
        D.CORRECTION = {"cube": None, "cible": None}
        rows = []
        for k in labels:
            img_c = cam.corriger(cv2.imread(os.path.join(HERE, "essais_img", f"{k}.png")))
            r, _ = D.detecter(img_c, cam, seuils)
            c = r.get("cube")
            if c is None:
                print(f"  {meth} {k} : cube non détecté")
                continue
            rows.append((k, np.array(c["xyz"][:2]), V[k][0], c.get("lacet_deg", 0.0), V[k][1]))
        res[meth] = rows

    print(f"\n{len(labels)} essais. Vérité = pince du robot.\n")
    print(f"  {'essai':6s} " + "  ".join(f"{m:>26s}" for m in res))
    for k in labels:
        cells = []
        for m, rows in res.items():
            r = next((x for x in rows if x[0] == k), None)
            cells.append("non détecté".rjust(26) if r is None else
                         f"{1000 * np.linalg.norm(r[1] - r[2]):6.1f} mm {ecart_mod90(r[3], r[4]):+6.1f}°".rjust(26))
        print(f"  {k:6s} " + "  ".join(cells))

    print("\nRésumé (position : erreur horizontale ; lacet : écart modulo 90°) :")
    bilan = {}
    for m, rows in res.items():
        ep = np.array([1000 * np.linalg.norm(r[1] - r[2]) for r in rows])
        el = np.array([abs(ecart_mod90(r[3], r[4])) for r in rows])
        # correction affine en validation croisée
        loo = []
        for r in rows:
            tr = [x for x in rows if x[0] != r[0]]
            A = np.array([[*x[1], 1.0] for x in tr])
            M = np.linalg.lstsq(A, np.array([x[2] for x in tr]), rcond=None)[0]
            loo.append(1000 * np.linalg.norm(np.r_[r[1], 1.0] @ M - r[2]))
        bilan[m] = (ep.mean(), np.mean(loo), rows)
        print(f"  {m:10s} position brute {ep.mean():5.1f} mm (max {ep.max():5.1f})"
              f" | corrigée (val. croisée) {np.mean(loo):5.1f} mm (max {np.max(loo):5.1f})"
              f" | lacet {el.mean():4.1f}° (max {el.max():4.1f}°)")

    path0 = os.path.join(HERE, "correction_detection.json")
    if os.path.exists(path0):
        c0 = json.load(open(path0)).get("cube")
        if c0 and c0.get("methode_detection") in bilan:
            M0 = np.array(c0["M"])
            e0 = [1000 * np.linalg.norm(np.r_[r[1], 1.0] @ M0 - r[2]) for r in bilan[c0["methode_detection"]][2]]
            print(f"  avec l'ANCIENNE correction ({c0['methode_detection']}) : {np.mean(e0):5.1f} mm (max {np.max(e0):5.1f})")

    best = min(bilan, key=lambda m: bilan[m][1])
    rows = bilan[best][2]
    A = np.array([[*x[1], 1.0] for x in rows])
    M = np.linalg.lstsq(A, np.array([x[2] for x in rows]), rcond=None)[0]
    path = os.path.join(HERE, "correction_detection.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    if "M" in old:                                   # ancien format : on garde la partie cible
        old = {"cible": {"M": old["M"]}}
    old["cube"] = {"M": M.round(6).tolist(), "methode_detection": best,
                   "erreur_validation_croisee_mm": round(float(bilan[best][1]), 2), "n_essais": len(rows)}
    old["formule"] = "(x, y)_corr = [x, y, 1] @ M, sur le plan de la table"
    with open(path, "w") as f:
        json.dump(old, f, indent=2)
    print(f"\nMéthode retenue pour le cube : '{best}' -> correction enregistrée dans correction_detection.json")
    if best == "centroide":
        print("detection.py utilise 'modele' par défaut : lance-le avec --centroide pour profiter de cette correction.")


if __name__ == "__main__":
    main()
