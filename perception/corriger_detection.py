"""Correction fine de la détection webcam à partir des essais mesurés au robot.

Les essais (detection_essais.json + verite_detection.txt, cf. erreur_detection.py) montrent une erreur
SYSTÉMATIQUE (biais vers la webcam). On apprend une petite correction affine 2D
    (x, y)_corrigé = M . (x, y, 1)_détecté
sur ces points, et on l'évalue honnêtement en VALIDATION CROISÉE « un essai laissé de côté » :
pour chaque essai, la correction est apprise SANS lui puis testée sur lui.
Comparé aussi : biais constant, et ré-estimation complète de la pose de la webcam (solvePnP).

Usage :  python perception/corriger_detection.py
Sortie : perception/correction_detection.json (appliquée automatiquement par detection.py).
À refaire si la webcam est recalée (recaler_webcam.py) ou déplacée.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def charger():
    ess = json.load(open(os.path.join(HERE, "detection_essais.json")))
    P = {}
    for line in open(os.path.join(HERE, "verite_detection.txt")):
        a = line.split()
        if len(a) >= 13:
            P[a[0]] = (np.array(a[1:4], float), np.array(a[4:13], float).reshape(3, 3))
    bl = [k for k in P if k.startswith("blanche")]
    d = np.mean([-P[k][0][2] / P[k][1][2, 2] for k in bl])
    pointe = {k: p + d * R[:, 2] for k, (p, R) in P.items()}
    rows = []
    for e, det in ess.items():
        for obj in ("cube", "cible"):
            k = f"{obj}_{e}"
            if det.get(obj) and k in pointe:
                rows.append((e, obj, np.array(det[obj]["xyz"][:2]), pointe[k][:2]))
    return ess, rows


def fit_affine(rows):
    A = np.array([[*r[2], 1.0] for r in rows])
    Y = np.array([r[3] for r in rows])
    return np.linalg.lstsq(A, Y, rcond=None)[0]          # 3 x 2


def fit_biais(rows):
    b = np.mean([r[3] - r[2] for r in rows], 0)
    return np.vstack([np.eye(2), b])


def appliquer(M, p):
    return np.r_[p, 1.0] @ M


def validation_croisee(rows, essais, fit):
    err = {"cube": [], "cible": []}
    for e in essais:
        M = fit([r for r in rows if r[0] != e])
        for r in rows:
            if r[0] == e:
                err[r[1]].append(1000 * np.linalg.norm(appliquer(M, r[2]) - r[3]))
    return err


def main():
    ess, rows = charger()
    essais = sorted(ess)
    if len(essais) < 4:
        raise SystemExit("Il faut au moins 4 essais.")
    essais = [e for e in essais if any(r[0] == e for r in rows)]   # seulement les essais pointés au crayon
    brut = {o: [1000 * np.linalg.norm(r[2] - r[3]) for r in rows if r[1] == o] for o in ("cube", "cible")}
    resultats = {"sans correction": brut,
                 "biais constant": validation_croisee(rows, essais, fit_biais),
                 "affine 2D": validation_croisee(rows, essais, fit_affine)}
    print(f"{len(essais)} essais, {len(rows)} mesures. Erreurs en VALIDATION CROISÉE (essai testé jamais appris) :\n")
    print(f"  {'méthode':18s} {'cube moy/max':>16s} {'cible moy/max':>16s}")
    for nom, err in resultats.items():
        print(f"  {nom:18s} {np.mean(err['cube']):7.1f} /{np.max(err['cube']):5.1f} mm"
              f" {np.mean(err['cible']):7.1f} /{np.max(err['cible']):5.1f} mm")
    M = fit_affine(rows)
    path = os.path.join(HERE, "correction_detection.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    if "M" in old:
        old = {}
    erreurs = {o: round(float(np.mean(v)), 2) for o, v in resultats["affine 2D"].items()}
    out = dict(old)
    out["cible"] = {"M": M.round(6).tolist(), "erreur_validation_croisee_mm": erreurs["cible"],
                    "n_essais": len(essais)}
    if "cube" not in old:              # ne remplace pas une correction « modele » apprise par evaluer_detection.py
        out["cube"] = {"M": M.round(6).tolist(), "methode_detection": "centroide",
                       "erreur_validation_croisee_mm": erreurs["cube"], "n_essais": len(essais)}
    out["formule"] = "(x, y)_corr = [x, y, 1] @ M, sur le plan de la table"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nCorrection affine (apprise sur tous les essais) -> perception/correction_detection.json")
    print("detection.py l'applique désormais automatiquement (--brut pour la désactiver).")


if __name__ == "__main__":
    main()
