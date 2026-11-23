"""Envoie les fichiers du dossier ubuntu/ vers le PC Ubuntu par le réseau (port 5006).
Sur l'Ubuntu, le petit programme de réception doit tourner AVANT de lancer celui-ci.

    python ubuntu/envoyer_ubuntu.py 10.254.79.65
"""
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FICHIERS = ["serveur_robot.py", "jumeau_robot.cpp"]

ip = sys.argv[1]
for nom in FICHIERS:
    with open(os.path.join(HERE, nom), "rb") as f:
        contenu = f.read()
    with socket.create_connection((ip, 5006), timeout=10) as s:
        s.sendall(nom.encode() + b"\n" + contenu)
    print(f"envoyé : {nom} ({len(contenu)} octets)")
