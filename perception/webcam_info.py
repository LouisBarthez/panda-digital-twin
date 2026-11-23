"""Premier test de la webcam : trouve son numero, ses resolutions, et l'affiche en direct.

    python perception\\webcam_info.py              -> cherche les cameras (numeros 0 a 5) et teste les resolutions
    python perception\\webcam_info.py --index 1    -> vue en direct de la camera 1
    python perception\\webcam_info.py --index 1 --res 1280x720   -> vue en direct dans une resolution donnee

Vue en direct :  s = enregistre une image dans perception/captures/   q = quitte
Remarque : la RealSense apparait aussi comme une camera (son flux couleur). Pour savoir laquelle est
la webcam, regarde la vue en direct.
Les parametres optiques de la webcam (fx, fy, cx, cy, distorsion) ne sont pas fournis par la camera :
il faut les mesurer avec un damier -> calib_webcam.py.
"""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
RESOLUTIONS = [(640, 480), (800, 600), (1280, 720), (1920, 1080)]


def open_cam(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)       # DirectShow : le plus fiable sous Windows
    return cap if cap.isOpened() else None


def scan():
    print("Recherche des cameras (numeros 0 a 5)...")
    found = []
    for idx in range(6):
        cap = open_cam(idx)
        if cap is None:
            continue
        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue
        print(f"\nCamera {idx} : image par defaut {frame.shape[1]}x{frame.shape[0]}")
        for w, h in RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, frame = cap.read()
            got = (frame.shape[1], frame.shape[0]) if ok else None
            fps = cap.get(cv2.CAP_PROP_FPS)
            status = "OK" if got == (w, h) else f"non (donne {got[0]}x{got[1]})" if got else "non"
            print(f"  {w}x{h} : {status}   (fps annonce : {fps:.0f})")
        af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
        print(f"  autofocus (propriete OpenCV) : {af}  (-1 ou 0 souvent = non gere / pas d'autofocus)")
        cap.release()
        found.append(idx)
    if not found:
        print("Aucune camera trouvee.")
    else:
        print(f"\nCameras trouvees : {found}")
        print("Vue en direct : python perception\\webcam_info.py --index N")


def live(index, res):
    cap = open_cam(index)
    if cap is None:
        raise SystemExit(f"Impossible d'ouvrir la camera {index}")
    if res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
    os.makedirs(os.path.join(HERE, "captures"), exist_ok=True)
    t0, n = time.time(), 0
    fps_txt = ""
    print("Vue en direct : s = enregistrer, q = quitter")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Plus d'image : camera debranchee ?")
            break
        n += 1
        if time.time() - t0 > 1:
            fps_txt = f"{n / (time.time() - t0):.0f} img/s"
            t0, n = time.time(), 0
        shown = frame.copy()
        cv2.putText(shown, f"camera {index}  {frame.shape[1]}x{frame.shape[0]}  {fps_txt}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Webcam", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            name = os.path.join("captures", f"webcam_{time.strftime('%H%M%S')}.png")
            cv2.imwrite(os.path.join(HERE, name), frame)
            print(f"Enregistre : {name}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--index" in sys.argv:
        idx = int(sys.argv[sys.argv.index("--index") + 1])
        res = None
        if "--res" in sys.argv:
            w, h = sys.argv[sys.argv.index("--res") + 1].lower().split("x")
            res = (int(w), int(h))
        live(idx, res)
    else:
        scan()
