"""
Pose de la webcam par AprilTag fixe (tag 0) + pose relative d'un tag 1.
Adapté du script du tuteur (apriltag2.py) au setup de Louis :
  - webcam = caméra 1 (DirectShow), en 1280x720 (résolution de la calibration)
  - lit perception/webcam_intrinsics.json (ou un YAML ROS / K-D)
  - marche avec OpenCV 4.6 ET OpenCV >= 4.7 (l'API aruco a changé en 4.7)
  - recalage : --save-ref enregistre la pose du tag 0 (moyenne sur --n images) ;
    ensuite le script compare la pose MOYENNE (glissante) à la référence et
    n'alerte qu'au-delà de --seuil-mm (10 mm) ou --seuil-deg (1°).
    Affiche aussi le bruit de mesure (écart-type de la position de la caméra).

Usage (PowerShell, dans C:\\Users\\louis\\pfe_mujoco) :
    python perception/apriltag_pose.py --tag-size 0.100
    python perception/apriltag_pose.py --tag-size 0.100 --save-ref
Touches : q = quitter, r = enregistrer la référence (comme --save-ref)

--tag-size = côté du CARRÉ NOIR extérieur, mesuré au réglet sur le tag imprimé (en m).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_PATH = os.path.join(HERE, "webcam_tag0_ref.json")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def _find(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def _arr(node):
    if isinstance(node, dict) and "data" in node:
        node = node["data"]
    return np.array(node, dtype=np.float64)


def load_calibration(path):
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
    else:
        with open(path) as f:
            data = json.load(f)

    Kn = _find(data, ["camera_matrix", "K", "mtx", "intrinsic_matrix"])
    if Kn is not None:
        K = _arr(Kn).reshape(3, 3)
    elif all(k in data for k in ("fx", "fy", "cx", "cy")):
        K = np.array([[data["fx"], 0, data["cx"]],
                      [0, data["fy"], data["cy"]],
                      [0, 0, 1]], dtype=np.float64)
    else:
        raise KeyError(f"matrice K introuvable. Clés du fichier : {list(data.keys())}")

    Dn = _find(data, ["distortion_coefficients", "dist_coefficients", "dist_coeffs",
                      "dist", "D", "distortion"])
    if Dn is None:
        # n'importe quelle clé contenant "dist" (ex. "dist_coeffs_k3fixed")
        for k, v in data.items():
            if "dist" in k.lower() and isinstance(v, (list, dict)):
                Dn = v
                print(f"[INFO] distorsion lue dans la clé '{k}'")
                break
    if Dn is None and all(k in data for k in ("k1", "k2", "p1", "p2")):
        Dn = [data["k1"], data["k2"], data["p1"], data["p2"], data.get("k3", 0.0)]
    if Dn is None:
        print("[ERREUR] distorsion introuvable dans le fichier de calibration.")
        print(f"         Clés présentes : {list(data.keys())}")
        print("         -> vérifier le fichier de calibration. Arrêt (la pose serait biaisée).")
        sys.exit(1)
    D = _arr(Dn).flatten()

    res = _find(data, ["resolution", "image_size", "size"])
    return K, D, res


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------
def to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec).flatten()
    return T


def estimate_tag_pose(corners, tag_size, K, D):
    h = tag_size / 2.0
    # ordre des coins OpenCV : haut-gauche, haut-droit, bas-droit, bas-gauche
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 2).astype(np.float64),
                                  K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else (None, None)


def rot_angle_deg(R):
    c = (np.trace(R) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def mean_pose(Ts):
    """Moyenne de poses : translation moyenne, rotation moyenne projetée sur SO(3)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    M = np.sum([T[:3, :3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_detector(dict_id):
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    if hasattr(cv2.aruco, "ArucoDetector"):              # OpenCV >= 4.7
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        return lambda g: det.detectMarkers(g)
    p = cv2.aruco.DetectorParameters_create()            # OpenCV 4.6
    return lambda g: cv2.aruco.detectMarkers(g, d, parameters=p)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=os.path.join(HERE, "webcam_intrinsics.json"))
    ap.add_argument("--tag-size", type=float, required=True,
                    help="côté du carré noir extérieur en mètres (mesuré)")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--res", default="1280x720")
    ap.add_argument("--family", default="tag36h11",
                    choices=["tag36h11", "tag25h9", "tag16h5"])
    ap.add_argument("--save-ref", action="store_true",
                    help="enregistre la pose du tag 0 comme référence (moyenne sur --n images)")
    ap.add_argument("--n", type=int, default=60,
                    help="nombre d'images moyennées pour la référence et l'affichage")
    ap.add_argument("--seuil-mm", type=float, default=10.0)
    ap.add_argument("--seuil-deg", type=float, default=1.0)
    args = ap.parse_args()

    K, D, res_calib = load_calibration(args.calib)
    W, H = map(int, args.res.lower().split("x"))
    print(f"[INFO] OpenCV {cv2.__version__}")
    print(f"[INFO] K =\n{K}\n[INFO] D = {D}")
    if res_calib is not None:
        print(f"[INFO] résolution de la calibration (fichier) : {res_calib}")

    fam = {"tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
           "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
           "tag16h5": cv2.aruco.DICT_APRILTAG_16h5}[args.family]
    detect = make_detector(fam)

    ref = None
    if os.path.exists(REF_PATH):
        with open(REF_PATH) as f:
            ref = np.array(json.load(f)["T_cam_tag0"])
        print(f"[INFO] référence chargée : {REF_PATH}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    if not cap.isOpened():
        print(f"[ERREUR] caméra {args.camera} impossible à ouvrir")
        sys.exit(1)
    ok, frame = cap.read()
    if not ok or frame.shape[1] != W or frame.shape[0] != H:
        got = None if not ok else f"{frame.shape[1]}x{frame.shape[0]}"
        print(f"[ERREUR] la caméra donne {got} au lieu de {W}x{H} : "
              f"la calibration ne serait pas valable.")
        sys.exit(1)
    print("[INFO] q = quitter, r = enregistrer la référence")

    save_next = args.save_ref
    hist = []
    last_print = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERREUR] lecture impossible")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        poses = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, tid in enumerate(ids.flatten()):
                if tid not in (0, 1):
                    continue
                rvec, tvec = estimate_tag_pose(corners[i], args.tag_size, K, D)
                if rvec is None:
                    continue
                poses[int(tid)] = to_T(rvec, tvec)
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, args.tag_size * 0.5)
                c = corners[i][0].mean(axis=0).astype(int)
                cv2.putText(frame, f"ID{tid} d={np.linalg.norm(tvec):.3f}m",
                            (c[0] - 50, c[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        y = 28
        if 0 in poses:
            hist.append(poses[0])
            if len(hist) > args.n:
                hist.pop(0)
            T0m = mean_pose(hist)
            if save_next and len(hist) >= args.n:
                with open(REF_PATH, "w") as f:
                    json.dump({"T_cam_tag0": T0m.tolist(), "tag_size": args.tag_size,
                               "family": args.family, "res": args.res, "n": len(hist)},
                              f, indent=2)
                ref = T0m
                save_next = False
                print(f"[INFO] référence enregistrée (moyenne de {len(hist)} images) -> {REF_PATH}")
            elif save_next:
                cv2.putText(frame, f"mesure de la reference... {len(hist)}/{args.n}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                y += 28
            # bruit : écart-type de la position de la caméra sur l'historique
            if len(hist) >= 10:
                P = np.array([np.linalg.inv(T)[:3, 3] for T in hist])
                bruit = 1000 * np.linalg.norm(P.std(axis=0))
                cv2.putText(frame, f"bruit position camera : {bruit:.1f} mm ({len(hist)} img)",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 28
            if ref is not None and len(hist) >= min(10, args.n):
                Tc_now = np.linalg.inv(T0m)
                Tc_ref = np.linalg.inv(ref)
                dpos = 1000 * np.linalg.norm(Tc_now[:3, 3] - Tc_ref[:3, 3])
                dang = rot_angle_deg(Tc_ref[:3, :3].T @ Tc_now[:3, :3])
                moved = dpos > args.seuil_mm or dang > args.seuil_deg
                col = (0, 0, 255) if moved else (0, 200, 0)
                txt = (f"camera vs ref (moyenne) : {dpos:.1f} mm, {dang:.2f} deg"
                       + ("  -> A RECALER" if moved else "  OK"))
                cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
                y += 28
        else:
            cv2.putText(frame, "tag 0 non vu", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
            y += 28

        if 0 in poses and 1 in poses:
            T01 = np.linalg.inv(poses[0]) @ poses[1]
            t = T01[:3, 3]
            cv2.putText(frame, f"tag1 / tag0 : [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            key = tuple(np.round(t, 3))
            if key != last_print:
                print(f"tag1/tag0 t = {t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f} m, "
                      f"|t| = {np.linalg.norm(t):.4f} m")
                last_print = key

        cv2.imshow("AprilTag", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            save_next = True
            hist = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
