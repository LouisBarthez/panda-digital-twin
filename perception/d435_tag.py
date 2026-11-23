"""Calibration de la D435 au poignet — côté caméra (PC Windows).

À chaque pose du robot (immobile, tag 0 bien visible par la D435) :
  - sur l'Ubuntu : lire_pose --libre enregistre la pose du robot avec une étiquette (p1, p2...) ;
  - ici : ESPACE enregistre la pose du tag vue par la D435 (moyenne de 30 images) avec LA MÊME étiquette.

Usage (PowerShell, à la racine du dépôt) :
    python perception/d435_tag.py --tag-size 0.080
Touches dans la fenêtre : ESPACE = enregistrer (l'étiquette se tape dans la console), q = quitter.
Sorties : perception/d435_tags.json (une entrée par étiquette) + perception/d435_calib/<etiquette>.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from apriltag_pose import estimate_tag_pose, make_detector, mean_pose, to_T  # noqa: E402

OUT_JSON = os.path.join(HERE, "d435_tags.json")
OUT_DIR = os.path.join(HERE, "d435_calib")
MODES = [(1280, 720, 30), (848, 480, 30), (640, 480, 30), (640, 480, 15)]
N_MOY = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-size", type=float, required=True)
    args = ap.parse_args()

    import pyrealsense2 as rs
    pipe = rs.pipeline()
    prof = None
    for (w, h, fps) in MODES:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        try:
            prof = pipe.start(cfg)
            print(f"[d435] mode couleur : {w}x{h} à {fps} img/s")
            break
        except RuntimeError:
            pass
    if prof is None:
        raise SystemExit("[ERREUR] la D435 ne démarre pas (appli Caméra / RealSense Viewer ouverts ?)")
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    D = np.array(intr.coeffs[:5], dtype=np.float64)
    W, H = intr.width, intr.height
    print(f"[d435] intrinsèques d'usine : fx={intr.fx:.2f} fy={intr.fy:.2f} cx={intr.ppx:.2f} cy={intr.ppy:.2f}")

    detect = make_detector(cv2.aruco.DICT_APRILTAG_36h11)
    data = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            data = json.load(f)
        print(f"[info] {len(data)} poses déjà enregistrées dans d435_tags.json : {list(data)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    hist = []
    print("ESPACE = enregistrer cette pose ; q = quitter")
    try:
        while True:
            frame = np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
            corners, ids, _ = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            T0 = None
            if ids is not None and 0 in ids.flatten():
                c = corners[list(ids.flatten()).index(0)]
                rvec, tvec = estimate_tag_pose(c, args.tag_size, K, D)
                if rvec is not None:
                    T0 = to_T(rvec, tvec)
                    vis = frame.copy()
                    cv2.aruco.drawDetectedMarkers(vis, [c], np.array([[0]]))
                    cv2.drawFrameAxes(vis, K, D, rvec, tvec, args.tag_size * 0.5)
                    cv2.putText(vis, f"tag 0 a {np.linalg.norm(tvec):.3f} m", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if T0 is None:
                vis = frame.copy()
                cv2.putText(vis, "tag 0 non vu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                hist = []
            else:
                hist = (hist + [T0])[-N_MOY:]
            cv2.putText(vis, f"{len(data)} poses enregistrees  |  ESPACE = enregistrer, q = quitter",
                        (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("D435 - calibration main-oeil", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                if len(hist) < N_MOY:
                    print(f"  pas assez d'images stables du tag ({len(hist)}/{N_MOY}), attends une seconde")
                    continue
                label = input("  Étiquette (la même que sur l'Ubuntu, ex. p1) : ").strip()
                if not label:
                    continue
                Tm = mean_pose(hist)
                P = np.array([np.linalg.inv(T)[:3, 3] for T in hist])
                data[label] = {"T_cam_tag0": Tm.tolist(), "bruit_mm": float(1000 * np.linalg.norm(P.std(0))),
                               "K": K.tolist(), "D": D.tolist(), "res": [W, H], "tag_size": args.tag_size}
                cv2.imwrite(os.path.join(OUT_DIR, f"{label}.png"), frame)
                with open(OUT_JSON, "w") as f:
                    json.dump(data, f, indent=1)
                print(f"  -> {label} enregistré (tag à {np.linalg.norm(Tm[:3, 3]):.3f} m, "
                      f"bruit {data[label]['bruit_mm']:.1f} mm). Total : {len(data)}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
