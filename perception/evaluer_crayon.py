"""Précision de la webcam sur le cube ET la cible, vérité = crayon serré dans la pince (26/11).

Pour chaque essai dN (detection.py : ESPACE + étiquette dN -> image brute essais_img/dN.png), on touche
au crayon (lire_pose <IP> --libre, crayon serré au début) :
    cube_dN_a, cube_dN_b : deux coins OPPOSÉS (en diagonale) du DESSUS du cube
                           -> centre = milieu des deux coins, lacet = direction de la diagonale - 45°
    cible_dN             : centre de la cible
et une fois dans la série : blanche1 (pointe posée sur la table blanche, z = 0 -> longueur du crayon).

Les images sont re-détectées ici SANS correction (--brut), avec la pose actuelle de la webcam, puis
on apprend une correction par objet, décalage constant ou affine, la meilleure en validation croisée ( : chaque essai est prédit par une
correction apprise sans lui). La correction est écrite dans correction_detection.json (cube :
méthode « modele », cible), l'ancienne est archivée dans archives_calib/.

Usage :  python perception/evaluer_crayon.py --verite verite_crayon_d.txt [--serie d] [--sans-ecrire]
"""
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


def lire_points(path):
    """Lignes « étiquette x y z r11..r33 [...] » de lire_pose (les autres lignes sont ignorées)."""
    P = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        a = line.replace(",", " ").split()
        if len(a) < 13:
            continue
        try:
            v = np.array(a[1:13], float)
        except ValueError:
            continue
        P[a[0]] = (v[:3], v[3:12].reshape(3, 3))
    return P


def ecart_mod90(a, b):
    return (a - b + 45.0) % 90.0 - 45.0


def fit_affine(src, dst):
    A = np.c_[src, np.ones(len(src))]
    return np.linalg.lstsq(A, dst, rcond=None)[0]


def main():
    verite = _arg("--verite", "verite_crayon_d.txt")
    serie = _arg("--serie", "d")
    P = lire_points(os.path.join(HERE, verite))
    bl = [k for k in P if k.startswith("blanche")]
    if not bl:
        raise SystemExit("Il faut au moins un point 'blanche1' (pointe sur la table blanche) pour la longueur du crayon.")
    d = float(np.mean([-P[k][0][2] / P[k][1][2, 2] for k in bl]))
    pointe = {k: P[k][0] + d * P[k][1][:, 2] for k in P}
    incl = {k: np.degrees(np.arccos(min(1.0, abs(P[k][1][2, 2])))) for k in P}
    print(f"Longueur EE -> pointe du crayon : {d * 1000:.1f} mm (sur {len(bl)} point(s) blanche)")
    print(f"Inclinaison du crayon : max {max(incl.values()):.1f}° (corrigée par l'orientation enregistrée)\n")

    sys.argv = [sys.argv[0], "--brut"]
    import detection as D
    from twin_scene import red_z
    D.METHODE = "modele"
    D.CORRECTION = {"cube": None, "cible": None}
    cam = D.Webcam()
    seuils = D.charger_seuils()

    essais = sorted({k.split("_")[1] for k in pointe if k.startswith(("cube_", "cible_"))
                     and k.split("_")[1].startswith(serie)}, key=lambda e: (len(e), e))
    rows = {"cube": [], "cible": []}
    print(f"  {'essai':6s} {'cube : erreur brute':>22s} {'lacet':>8s} {'h dessus':>9s}   {'cible : erreur brute':>21s}")
    for e in essais:
        img_p = os.path.join(HERE, "essais_img", f"{e}.png")
        if not os.path.exists(img_p):
            print(f"  {e:6s} pas d'image essais_img/{e}.png")
            continue
        r, _ = D.detecter(cam.corriger(cv2.imread(img_p)), cam, seuils)
        ligne = f"  {e:6s} "
        ka, kb = f"cube_{e}_a", f"cube_{e}_b"
        if ka in pointe and kb in pointe and r.get("cube"):
            a, b = pointe[ka], pointe[kb]
            centre = (a[:2] + b[:2]) / 2
            diag = np.linalg.norm(a[:2] - b[:2]) * 1000
            lac_v = (np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) - 45.0) % 90.0
            h = (a[2] + b[2]) / 2 - red_z(*centre)
            mes = np.array(r["cube"]["xyz"][:2])
            el = ecart_mod90(r["cube"].get("lacet_deg", 0.0), lac_v)
            rows["cube"].append((e, mes, centre))
            ligne += f"{np.linalg.norm(mes - centre) * 1000:15.1f} mm   {el:+6.1f}° {h * 1000:6.1f} mm"
            if abs(diag - 56.6) > 6:
                ligne += f"  [diagonale {diag:.0f} mm au lieu de ~56 : coins bien opposés ?]"
        else:
            ligne += f"{'(pas de mesure)':>22s} {'':8s} {'':9s}"
        kc = f"cible_{e}"
        if kc in pointe and r.get("cible"):
            mes = np.array(r["cible"]["xyz"][:2])
            rows["cible"].append((e, mes, pointe[kc][:2]))
            ligne += f"   {np.linalg.norm(mes - pointe[kc][:2]) * 1000:14.1f} mm"
        print(ligne)

    print("\nRésumé (erreur horizontale) :")
    corr = {}
    path = os.path.join(HERE, "correction_detection.json")
    ancien = json.load(open(path)) if os.path.exists(path) else {}
    for obj, R in rows.items():
        if len(R) < 4:
            print(f"  {obj:5s} : {len(R)} essai(s), il en faut au moins 4")
            continue
        mes = np.array([x[1] for x in R])
        vrai = np.array([x[2] for x in R])
        brut = np.linalg.norm(mes - vrai, axis=1) * 1000
        biais = (mes - vrai).mean(0) * 1000
        txt = f"  {obj:5s} ({len(R)} essais) brute {brut.mean():4.1f} mm (max {brut.max():4.1f}, biais dx {biais[0]:+.1f} dy {biais[1]:+.1f})"
        a0 = ancien.get(obj)
        if a0 and (obj == "cible" or a0.get("methode_detection") == "modele"):
            M0 = np.array(a0["M"])
            e0 = np.linalg.norm(np.c_[mes, np.ones(len(mes))] @ M0 - vrai, axis=1) * 1000
            txt += f" | ancienne corr. {e0.mean():4.1f} mm (max {e0.max():4.1f})"
        # deux corrections candidates, jugées en validation croisée : décalage constant (2 paramètres,
        # robuste avec peu d'essais) et affine (6 paramètres, corrige aussi échelle / rotation)
        def fit_biais(src, dst):
            M = np.vstack([np.eye(2), (dst - src).mean(0)])
            return M
        res_c = {}
        for nom, fit in (("decalage", fit_biais), ("affine", fit_affine)):
            loo = []
            for i in range(len(R)):
                m = np.arange(len(R)) != i
                M = fit(mes[m], vrai[m])
                loo.append(np.linalg.norm(np.r_[mes[i], 1.0] @ M - vrai[i]) * 1000)
            res_c[nom] = (fit(mes, vrai), float(np.mean(loo)), float(np.max(loo)))
            txt += f" | {nom} (val. croisée) {np.mean(loo):4.1f} mm (max {np.max(loo):4.1f})"
        print(txt)
        choix = min(res_c, key=lambda k: res_c[k][1])
        corr[obj] = (res_c[choix][0], res_c[choix][1], len(R), float(brut.mean()), choix)

    if not corr or "--sans-ecrire" in sys.argv:
        return
    os.makedirs(os.path.join(HERE, "archives_calib"), exist_ok=True)
    if os.path.exists(path):
        shutil.copy(path, os.path.join(HERE, "archives_calib", time.strftime("%Y-%m-%d_%H-%M-%S_") + "correction_detection.json"))
    for obj, (M, eloo, n, eb, choix) in corr.items():
        if eloo >= eb:
            print(f"  {obj} : la correction n'améliore pas l'erreur brute -> pas de correction")
            ancien[obj] = None
            continue
        print(f"  {obj} : correction « {choix} » retenue ({eb:.1f} -> {eloo:.1f} mm en validation croisée)")
        ancien[obj] = {"M": M.round(6).tolist(), "type": choix, "erreur_validation_croisee_mm": round(eloo, 2),
                       "n_essais": n,
                       "verite": f"crayon ({verite})", "date": time.strftime("%Y-%m-%d %H:%M")}
        if obj == "cube":
            ancien[obj]["methode_detection"] = "modele"
    ancien["formule"] = "(x, y)_corr = [x, y, 1] @ M, sur le plan de la table"
    with open(path, "w") as f:
        json.dump(ancien, f, indent=2)
    print("\n-> correction_detection.json mis à jour (ancien fichier dans archives_calib/)")


if __name__ == "__main__":
    main()
