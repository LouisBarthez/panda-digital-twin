"""Calibration optique de la webcam avec un damier 9x6 coins : fx, fy, cx, cy et distorsion.

CAPTURE (les images vont dans perception/<dossier>/ ; seule cette serie est effacee au depart) :
    python perception\\calib_webcam.py --index 1 --res 1280x720 --out calib_webcam_petit
    ESPACE = garder l'image (si le damier est detecte), c ou q = terminer.
    --no-calib : seulement capturer, sans calibrer a la fin.

CALIBRATION a partir d'images deja prises (une ou plusieurs series) :
    python perception\\calib_webcam.py --from-images --dirs calib_webcam_serie1:25,calib_webcam:25,calib_webcam_petit:12.5
    Chaque serie = dossier:taille_de_case_en_mm. (La taille de case ne change pas fx, fy, cx, cy ni la
    distorsion, seulement les distances affichees : on peut melanger un damier imprime a 100 % et a 50 %.)
    --fix-k3 : bloque k3 a 0 (conseille quand les vues ne couvrent pas bien les bords de l'image).
    Pour retirer une image ratee : supprimer son fichier, puis relancer.

VERIFICATIONS AFFICHEES :
    - erreur de reprojection par image et globale (< 0.5 px tres bien, < 1 px correct) ;
    - test de rectitude : les coins d'une meme rangee du damier doivent etre alignes. On mesure leur ecart
      a une droite, sur l'image brute puis corrigee : si la correction le fait chuter, la distorsion est reelle
      et bien corrigee ;
    - deplacement des pixels par la correction, au centre des bords et dans les coins de l'image.
Sorties : perception/webcam_intrinsics.json, perception/<1re serie>/verif_avant_apres.png
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
COLS, ROWS = 9, 6            # coins INTERIEURS du damier (10 x 7 cases)
MIN_IMAGES = 15
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def board_points(case_m):
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * case_m
    return objp


def find_corners(gray):
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS),
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRIT)
    return found, corners


def capture(index, w, h, out_name):
    out_dir = os.path.join(HERE, out_name)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "calib_*.png")):
        os.remove(old)
    n = 0
    print(f"Images enregistrees dans perception/{out_name}/ ; ESPACE = garder, c ou q = terminer")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Plus d'image de la camera")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        shown = frame.copy()
        if found:
            cv2.drawChessboardCorners(shown, (COLS, ROWS), corners, found)
        cv2.putText(shown, f"images gardees : {n}  {'damier OK' if found else 'damier non detecte'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("Calibration webcam", shown)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("c"), ord("q")):
            break
        if key == ord(" ") and found:
            n += 1
            cv2.imwrite(os.path.join(out_dir, f"calib_{n:02d}.png"), frame)
            print(f"image {n} gardee")
    cap.release()
    cv2.destroyAllWindows()
    return n


def straightness(pts):
    """Ecart moyen (px) des coins a la droite ajustee sur chaque rangee et chaque colonne du damier."""
    g = pts.reshape(ROWS, COLS, 2)
    lines = [g[r] for r in range(ROWS)] + [g[:, c] for c in range(COLS)]
    devs = []
    for L in lines:
        m = L.mean(0)
        _, _, vt = np.linalg.svd(L - m)
        normal = vt[1]
        devs.append(np.abs((L - m) @ normal).max())
    return float(np.mean(devs))


def calibrate(series, fix_k3):
    obj_pts, img_pts, names = [], [], []
    size, first_file = None, None
    for folder, case_m in series:
        files = sorted(glob.glob(os.path.join(HERE, folder, "calib_*.png")))
        print(f"Serie {folder} (case {case_m * 1000:.1f} mm) : {len(files)} images")
        for f in files:
            gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
            s = (gray.shape[1], gray.shape[0])
            if size is None:
                size, first_file = s, f
            elif s != size:
                print(f"  {os.path.basename(f)} : taille {s} differente de {size}, ignoree")
                continue
            found, corners = find_corners(gray)
            if not found:
                print(f"  damier non retrouve dans {folder}/{os.path.basename(f)} : ignoree")
                continue
            obj_pts.append(board_points(case_m))
            img_pts.append(corners)
            names.append(f"{folder}/{os.path.basename(f)}")
    if len(obj_pts) < MIN_IMAGES:
        raise SystemExit(f"Seulement {len(obj_pts)} images exploitables : il en faut au moins {MIN_IMAGES}.")
    w, h = size

    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    print(f"\nCalibration sur {len(obj_pts)} images {w}x{h}{'  (k3 bloque a 0)' if fix_k3 else ''}...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (w, h), None, None, flags=flags)

    print("\nPar image : erreur de reprojection | distance | rectitude brute -> corrigee (px)")
    raw_s, cor_s = [], []
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        e = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts[i].reshape(-1, 2)) ** 2, axis=1))))
        und = cv2.undistortPoints(img_pts[i].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        rs, cs = straightness(img_pts[i].reshape(-1, 2)), straightness(und)
        raw_s.append(rs)
        cor_s.append(cs)
        flag = "   <- a refaire" if e > 1.0 else ""
        print(f"  {names[i]:34s} {e:.3f} px  {np.linalg.norm(tvecs[i]) * 100:5.1f} cm   "
              f"{rs:.2f} -> {cs:.2f}{flag}")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fovx = np.degrees(2 * np.arctan(w / (2 * fx)))
    fovy = np.degrees(2 * np.arctan(h / (2 * fy)))
    print(f"\nErreur de reprojection globale : {rms:.3f} pixel")
    print(f"fx = {fx:.2f}  fy = {fy:.2f}  cx = {cx:.2f}  cy = {cy:.2f}")
    print(f"distorsion (k1 k2 p1 p2 k3) = {np.round(dist.ravel(), 4).tolist()}")
    print(f"champ : {fovx:.1f} x {fovy:.1f} degres  (fovy MuJoCo = {fovy:.2f})")

    print(f"\nTest de rectitude (ecart moyen des coins a une droite) : brut {np.mean(raw_s):.2f} px"
          f"  ->  corrige {np.mean(cor_s):.2f} px")
    print("  brut nettement > corrige : il y a bien de la distorsion, et elle est corrigee.")
    print("  brut ~ corrige et faible (< 0.5 px) : distorsion negligeable.")

    # deplacement des pixels par la correction, en des points cles de l'image
    pts = {"centre": (w / 2, h / 2), "milieu bord gauche": (0, h / 2), "milieu bord haut": (w / 2, 0),
           "coin haut gauche": (0, 0), "coin bas droit": (w - 1, h - 1)}
    arr = np.array(list(pts.values()), np.float32).reshape(-1, 1, 2)
    moved = cv2.undistortPoints(arr, K, dist, P=K).reshape(-1, 2)
    print("\nDeplacement d'un pixel par la correction :")
    for (nm, p), q in zip(pts.items(), moved):
        print(f"  {nm:20s} {np.linalg.norm(q - np.array(p)):6.1f} px")
    print("  (dans les coins, une valeur de plusieurs dizaines de px veut dire une distorsion visible a l'oeil)")

    out = {"largeur": w, "hauteur": h, "series": [f"{f}:{c * 1000:g}" for f, c in series],
           "n_images": len(obj_pts), "k3_bloque": fix_k3, "erreur_reprojection_px": float(rms),
           "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
           "distorsion_k1k2p1p2k3": dist.ravel().tolist(),
           "champ_horizontal_deg": float(fovx), "champ_vertical_deg": float(fovy),
           "rectitude_brute_px": float(np.mean(raw_s)), "rectitude_corrigee_px": float(np.mean(cor_s))}
    with open(os.path.join(HERE, "webcam_intrinsics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nEcrit : perception/webcam_intrinsics.json")

    img = cv2.imread(first_file)
    und = cv2.undistort(img, K, dist)
    for im in (img, und):                      # grille de droites pour juger a l'oeil
        for x in range(0, w, w // 8):
            cv2.line(im, (x, 0), (x, h), (0, 255, 255), 1)
        for y in range(0, h, h // 6):
            cv2.line(im, (0, y), (w, y), (0, 255, 255), 1)
    path = os.path.join(os.path.dirname(first_file), "verif_avant_apres.png")
    cv2.imwrite(path, np.hstack([img, und]))
    print(f"Verification : {os.path.relpath(path, HERE)} (gauche brute, droite corrigee, grille jaune droite)")


def parse_series(text, default_case):
    series = []
    for item in text.split(","):
        if ":" in item:
            name, c = item.split(":")
            series.append((name.strip(), float(c) / 1000.0))
        else:
            series.append((item.strip(), default_case))
    return series


def main():
    case_m = float(arg("--case", 25)) / 1000.0
    fix_k3 = "--fix-k3" in sys.argv
    if "--from-images" in sys.argv:
        return calibrate(parse_series(arg("--dirs", "calib_webcam"), case_m), fix_k3)
    index = int(arg("--index", 0))
    w, h = (int(v) for v in arg("--res", "1280x720").lower().split("x"))
    out_name = arg("--out", "calib_webcam")
    n = capture(index, w, h, out_name)
    if "--no-calib" in sys.argv:
        return
    if n < MIN_IMAGES:
        print(f"{n} images gardees : il en faut au moins {MIN_IMAGES} pour calibrer cette serie seule.")
        return
    calibrate([(out_name, case_m)], fix_k3)


if __name__ == "__main__":
    main()
