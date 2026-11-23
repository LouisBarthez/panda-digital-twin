"""Vue en direct de la webcam globale pour régler son cadrage.

Usage (PowerShell, dans C:\\Users\\louis\\pfe_mujoco) :
    python perception/apercu_webcam.py

Affiche la webcam (caméra 1, 1280x720) avec une grille en tiers :
  - base du robot dans la bande du HAUT
  - cible dans la bande du BAS
  - zone de travail à l'intérieur du cadre jaune (marge de 8 %)
Touches : ESPACE = enregistre perception/vue_webcam.png ; q = quitter
"""
import os
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vue_webcam.png")
W, H = 1280, 720

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
if not cap.isOpened():
    raise SystemExit("[ERREUR] caméra 1 impossible à ouvrir (appli Caméra fermée ?)")

print("ESPACE = enregistrer vue_webcam.png ; q = quitter")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[ERREUR] lecture impossible")
        break
    img = frame.copy()
    h, w = img.shape[:2]
    # tiers
    for k in (1, 2):
        cv2.line(img, (w * k // 3, 0), (w * k // 3, h), (255, 255, 255), 1)
        cv2.line(img, (0, h * k // 3), (w, h * k // 3), (255, 255, 255), 1)
    # cadre de marge
    mx, my = int(0.08 * w), int(0.08 * h)
    cv2.rectangle(img, (mx, my), (w - mx, h - my), (0, 255, 255), 2)
    cv2.putText(img, "base robot ici", (10, h // 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "cible ici", (10, 5 * h // 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"{w}x{h}  ESPACE=photo  q=quitter", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Cadrage webcam", img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord(" "):
        cv2.imwrite(OUT, frame)  # image brute, sans la grille
        print(f"enregistré -> {OUT}")

cap.release()
cv2.destroyAllWindows()
