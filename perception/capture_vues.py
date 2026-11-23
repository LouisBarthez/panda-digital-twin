"""Capture une image de chaque caméra pour vérifier les angles de vue.

Usage (PowerShell, dans C:\\Users\\louis\\pfe_mujoco, venv activé) :
    python perception/capture_vues.py
    python perception/capture_vues.py --no-webcam      (D435 seule)

Produit dans perception/ :
    vue_webcam.png            webcam globale (caméra 1, 1280x720)
    vue_d435_couleur.png      D435 couleur
    vue_d435_profondeur.png   D435 profondeur alignée sur la couleur (0 -> 1,5 m en couleurs)

Fermer l'appli Caméra de Windows avant : elle bloque les caméras.
"""
import os
import sys
import cv2
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720

# Modes D435 essayés dans l'ordre (le premier qui démarre est gardé)
MODES_D435 = [(1280, 720, 30), (1280, 720, 15), (848, 480, 30), (640, 480, 30), (640, 480, 15)]


def capture_webcam(index=1):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    ok, img = False, None
    for _ in range(40):  # laisse l'exposition auto se stabiliser
        ok, img = cap.read()
    cap.release()
    if not ok:
        print(f"[webcam] ERREUR : pas d'image sur la caméra {index}")
        return
    path = os.path.join(OUT, "vue_webcam.png")
    cv2.imwrite(path, img)
    print(f"[webcam] {img.shape[1]}x{img.shape[0]} -> {path}")


def info_d435(rs):
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[d435] ERREUR : aucune RealSense détectée (câble ? port ?)")
        return False
    for d in devs:
        usb = d.get_info(rs.camera_info.usb_type_descriptor) \
            if d.supports(rs.camera_info.usb_type_descriptor) else "?"
        print(f"[d435] {d.get_info(rs.camera_info.name)}, "
              f"série {d.get_info(rs.camera_info.serial_number)}, USB {usb}")
        if usb.startswith("2"):
            print("[d435] ATTENTION : branchée en USB 2 -> modes limités. "
                  "Utiliser un port/câble USB 3 (bleu).")
    return True


def capture_d435():
    try:
        import pyrealsense2 as rs
    except ImportError:
        print("[d435] ERREUR : pyrealsense2 non installé")
        return
    if not info_d435(rs):
        return

    pipe = rs.pipeline()
    prof = None
    for (w, h, fps) in MODES_D435:
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        try:
            prof = pipe.start(cfg)
            print(f"[d435] mode utilisé : {w}x{h} à {fps} img/s")
            break
        except RuntimeError as e:
            print(f"[d435] {w}x{h}@{fps} refusé : {e}")
    if prof is None:
        print("[d435] ERREUR : aucun mode n'a démarré. "
              "Fermer l'appli Caméra / RealSense Viewer, débrancher-rebrancher la D435.")
        return

    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    try:
        for _ in range(40):
            frames = pipe.wait_for_frames()
        frames = align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data()).copy()
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * scale
    finally:
        pipe.stop()

    h, w = depth.shape
    cv2.imwrite(os.path.join(OUT, "vue_d435_couleur.png"), color)

    d8 = np.clip(depth / 1.5 * 255, 0, 255).astype(np.uint8)
    dcol = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    dcol[depth == 0] = 0  # pixels sans profondeur en noir
    cv2.imwrite(os.path.join(OUT, "vue_d435_profondeur.png"), dcol)

    valid = depth > 0
    c = depth[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
    c = c[c > 0]
    print(f"[d435] images -> {OUT}")
    print(f"[d435] pixels sans profondeur : {100 * (1 - valid.mean()):.1f} %")
    if c.size:
        print(f"[d435] profondeur au centre : {np.median(c):.3f} m")
    if valid.any():
        print(f"[d435] profondeur min / médiane : "
              f"{depth[valid].min():.3f} / {np.median(depth[valid]):.3f} m")


if __name__ == "__main__":
    if "--no-webcam" not in sys.argv:
        capture_webcam()
    capture_d435()
