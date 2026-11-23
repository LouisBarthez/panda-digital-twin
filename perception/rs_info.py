"""Premier test de la RealSense : identite de la camera, parametres optiques, vue en direct.

    python perception\\rs_info.py

Affiche :
  - modele, numero de serie, firmware, type de port USB (il faut de l'USB 3 pour la profondeur en bonne resolution)
  - parametres optiques (intrinseques) des flux couleur et profondeur, echelle de profondeur
Puis ouvre une fenetre : image couleur | profondeur en fausses couleurs, avec la distance au pixel central.
    touche s : enregistre l'image couleur, la profondeur brute (.npy) et la profondeur coloree dans perception/captures/
    touche q : quitte
Les intrinseques sont aussi ecrites dans perception/realsense_intrinsics.json (pour la sim et la calibration).

Installation (une fois) : python -m pip install pyrealsense2 opencv-python numpy
"""
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HERE = os.path.dirname(os.path.abspath(__file__))
# resolutions essayees dans l'ordre (la premiere qui demarre est gardee)
CONFIGS = [(1280, 720, 30), (848, 480, 30), (640, 480, 30), (424, 240, 15)]


def device_info():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise SystemExit("Aucune RealSense detectee : verifie le cable (USB 3, port bleu) et rebranche.")
    info = {}
    for d in devs:
        get = lambda k: d.get_info(k) if d.supports(k) else "?"  # noqa: E731
        info = {
            "modele": get(rs.camera_info.name),
            "serie": get(rs.camera_info.serial_number),
            "firmware": get(rs.camera_info.firmware_version),
            "usb": get(rs.camera_info.usb_type_descriptor),
        }
        print("Camera trouvee :")
        for k, v in info.items():
            print(f"  {k:9s}: {v}")
        if info["usb"].startswith("2"):
            print("  ATTENTION : branchee en USB 2 -> resolution et cadence limitees. Utilise un port USB 3.")
    return info


def start_pipeline():
    for w, h, fps in CONFIGS:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            profile = pipe.start(cfg)
            print(f"\nFlux demarres en {w}x{h} a {fps} images/s")
            return pipe, profile, (w, h, fps)
        except RuntimeError:
            continue
    raise SystemExit("Impossible de demarrer couleur + profondeur, meme en basse resolution.")


def intr_dict(stream_profile):
    i = stream_profile.as_video_stream_profile().get_intrinsics()
    fovx = np.degrees(2 * np.arctan(i.width / (2 * i.fx)))
    fovy = np.degrees(2 * np.arctan(i.height / (2 * i.fy)))
    return {"largeur": i.width, "hauteur": i.height, "fx": i.fx, "fy": i.fy, "cx": i.ppx, "cy": i.ppy,
            "distorsion_modele": str(i.model), "distorsion_coeffs": list(i.coeffs),
            "champ_horizontal_deg": float(fovx), "champ_vertical_deg": float(fovy)}


def main():
    info = device_info()
    pipe, profile, (w, h, fps) = start_pipeline()
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()          # metres par unite de profondeur brute
        color_intr = intr_dict(profile.get_stream(rs.stream.color))
        depth_intr = intr_dict(profile.get_stream(rs.stream.depth))
        print(f"\nEchelle de profondeur : {depth_scale} m par unite")
        for name, d in [("couleur", color_intr), ("profondeur", depth_intr)]:
            print(f"\nIntrinseques {name} :")
            for k, v in d.items():
                print(f"  {k:22s}: {v}")
        print("\n(fovy pour MuJoCo = champ_vertical_deg du flux couleur)")

        out = {"camera": info, "resolution": [w, h, fps], "depth_scale": depth_scale,
               "couleur": color_intr, "profondeur": depth_intr,
               "note": "profondeur alignee sur la couleur dans ce script : utiliser les intrinseques couleur"}
        with open(os.path.join(HERE, "realsense_intrinsics.json"), "w") as f:
            json.dump(out, f, indent=2)
        print("Ecrit : perception/realsense_intrinsics.json")

        align = rs.align(rs.stream.color)                    # profondeur reprojetee pixel a pixel sur la couleur
        os.makedirs(os.path.join(HERE, "captures"), exist_ok=True)
        print("\nFenetre ouverte : s = enregistrer, q = quitter")
        while True:
            frames = align.process(pipe.wait_for_frames())
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                continue
            color = np.asanyarray(c.get_data())
            depth = np.asanyarray(d.get_data())
            depth_m = depth * depth_scale
            vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_m, alpha=255 / 2.0), cv2.COLORMAP_JET)  # 0-2 m
            u, v = color.shape[1] // 2, color.shape[0] // 2
            dist = d.get_distance(u, v)
            for img in (color, vis):
                cv2.drawMarker(img, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            txt = f"centre : {dist * 100:.1f} cm" if dist > 0 else "centre : pas de mesure"
            cv2.putText(color, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            both = np.hstack([color, vis])
            scale = 1280 / both.shape[1]
            cv2.imshow("RealSense : couleur | profondeur (0-2 m)", cv2.resize(both, None, fx=scale, fy=scale))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                stamp = time.strftime("%H%M%S")
                base = os.path.join(HERE, "captures", stamp)
                cv2.imwrite(base + "_couleur.png", color)
                cv2.imwrite(base + "_profondeur_vis.png", vis)
                np.save(base + "_profondeur_m.npy", depth_m.astype(np.float32))
                print(f"Enregistre : captures/{stamp}_*")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
