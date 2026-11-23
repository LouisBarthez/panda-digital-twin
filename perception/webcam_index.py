"""Trouve automatiquement la webcam globale (celle qui voit le tag 0) et retient son numéro.

Sous Windows, le numéro des caméras (0, 1, 2...) peut changer quand on débranche / rebranche
une webcam USB : la caméra 1 peut devenir celle de l'ordinateur portable. Ce module :
  - lit le numéro retenu dans perception/webcam_index.txt (1 par défaut) ;
  - en ligne de commande, essaie les caméras 0 à 4 et garde celle qui voit le tag 0 en 1280x720.

Usage :  python perception/webcam_index.py        -> cherche et enregistre le bon numéro
"""
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FICHIER = os.path.join(HERE, "webcam_index.txt")


def index_webcam():
    """Numéro de la webcam globale (à utiliser avec cv2.VideoCapture(i, cv2.CAP_DSHOW))."""
    try:
        with open(FICHIER) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 1


def chercher():
    dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dct, cv2.aruco.DetectorParameters())
        detect = det.detectMarkers
    else:
        detect = lambda g: cv2.aruco.detectMarkers(g, dct)  # noqa: E731
    trouve = None
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  caméra {i} : absente")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        img = None
        for _ in range(25):
            ok, img = cap.read()
        cap.release()
        if img is None:
            print(f"  caméra {i} : pas d'image")
            continue
        _, ids, _ = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        voit = ids is not None and 0 in ids.flatten()
        print(f"  caméra {i} : {img.shape[1]}x{img.shape[0]}, tag 0 {'VU' if voit else 'non vu'}")
        if voit and img.shape[:2] == (720, 1280) and trouve is None:
            trouve = i
    if trouve is None:
        print("\nAucune caméra ne voit le tag 0 en 1280x720 : la webcam Logitech est-elle branchée ?"
              " Le tag est-il dégagé (bras hors champ) ?")
        return
    with open(FICHIER, "w") as f:
        f.write(str(trouve))
    print(f"\nWebcam globale = caméra {trouve} -> enregistré dans perception/webcam_index.txt")


if __name__ == "__main__":
    chercher()
