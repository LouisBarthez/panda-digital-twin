#!/usr/bin/env python3
"""Serveur côté Ubuntu : reçoit les demandes du jumeau (PC Windows) et pilote le robot
avec le programme libfranka jumeau_robot (rien d'autre que la bibliothèque standard de Python).

    {"cmd": "etat"}                 -> pose actuelle du robot (q, O_T_EE, largeur de la pince)
    {"cmd": "executer", "traj": {}} -> rejoue la trajectoire validée en simulation ; pendant le
                                       mouvement, envoie au fil de l'eau {"q": [7]} (toutes les 40 ms),
                                       {"pince": w}, {"ligne": "..."} puis {"fin": {...}}

Lancement (hôte, mea@) :
    python3 ~/dependencies/jumeau/serveur_robot.py
    python3 ~/dependencies/jumeau/serveur_robot.py --confirmer   -> Entrée sur l'Ubuntu avant chaque mouvement
Ctrl+C pour arrêter. Chaque exécution est archivée dans ~/dependencies/jumeau/essais/.
"""
import datetime
import json
import os
import socket
import socketserver
import subprocess
import sys
import time

ROBOT_IP = "192.168.153.114"
PORT = 5005
EXE = os.path.expanduser("~/dependencies/libfranka-0.8.0/build/examples/jumeau_robot")
DOSSIER = os.path.dirname(os.path.abspath(__file__))
ARCHIVES = os.path.join(DOSSIER, "essais")
CONFIRMER = "--confirmer" in sys.argv

LIMITES = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
           (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
VITESSE_MAX = 0.3


def ip_wifi():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "?"


def verifier(traj):
    """Contrôles avant d'envoyer quoi que ce soit au robot. Renvoie le texte du fichier d'étapes."""
    v = float(traj["vitesse"])
    if not 0.0 < v <= VITESSE_MAX:
        raise ValueError(f"vitesse {v} hors de ]0 ; {VITESSE_MAX}]")
    lignes = [f"vitesse {v}"]

    def ok_q(q, nom):
        if len(q) != 7:
            raise ValueError(f"{nom} : il faut 7 angles")
        for i, (a, (lo, hi)) in enumerate(zip(q, LIMITES)):
            if not lo < a < hi:
                raise ValueError(f"{nom} : joint{i + 1} = {a:.3f} hors butées")
        return " ".join(f"{a:.6f}" for a in q)

    lignes.append("depart " + ok_q(traj["q_depart"], "depart"))
    for e in traj["etapes"]:
        t = e["type"]
        if t == "q":
            nom = e.get("nom", "q").replace(" ", "_")
            lignes.append(f"q {nom} " + ok_q(e["q"], nom))
        elif t == "ouvrir":
            lignes.append(f"ouvrir {min(float(e['largeur']), 0.08):.4f}")
        elif t == "saisir":
            f = float(e["force"])
            if not 5 <= f <= 40:
                raise ValueError(f"force de prise {f} N hors de [5 ; 40]")
            lignes.append(f"saisir {float(e['largeur']):.4f} {f:.1f}")
        else:
            raise ValueError(f"étape inconnue : {t}")
    return "\n".join(lignes) + "\n"


class Gestion(socketserver.StreamRequestHandler):
    def repondre(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())

    def handle(self):
        try:
            req = json.loads(self.rfile.readline().decode())
        except ValueError:
            return self.repondre({"erreur": "requête illisible"})
        cmd = req.get("cmd")
        if cmd == "etat":
            return self.etat()
        if cmd == "executer":
            return self.executer(req.get("traj", {}))
        self.repondre({"erreur": f"commande inconnue : {cmd}"})

    def etat(self):
        try:
            r = subprocess.run([EXE, ROBOT_IP, "etat"], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as ex:
            return self.repondre({"erreur": f"jumeau_robot : {ex}"})
        js = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not js:
            return self.repondre({"erreur": (r.stdout + r.stderr).strip()[-200:] or "pas de réponse du robot"})
        self.repondre(json.loads(js[-1]))

    def executer(self, traj):
        try:
            texte = verifier(traj)
        except (KeyError, TypeError, ValueError) as ex:
            print(f"[REFUS] trajectoire invalide : {ex}")
            return self.repondre({"fin": {"ok": False, "code": -2, "log": [f"refusée par le serveur : {ex}"], "mesures": []}})
        os.makedirs(ARCHIVES, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fichier = os.path.join(ARCHIVES, f"{stamp}_etapes.txt")
        with open(fichier, "w") as f:
            f.write(texte)
        info = traj.get("info", {})
        print(f"\n=== Trajectoire reçue ({stamp}) : {len(traj['etapes'])} étapes, vitesse {traj['vitesse']}")
        print(f"    cube {info.get('cube')}  lacet {info.get('lacet')}  cible {info.get('cible')}")
        print(f"    simulation : {info.get('bilan_sim')}")
        if CONFIRMER and input("    Entrée = lancer, n = refuser : ").strip().lower() == "n":
            return self.repondre({"fin": {"ok": False, "code": -3, "log": ["refusée sur l'Ubuntu"], "mesures": []}})
        t0 = time.time()
        log, mesures = [], []
        client_present = True

        def transmettre(obj):
            # envoi au Windows au fil de l'eau ; si le Windows a fermé, le robot finit quand même
            nonlocal client_present
            if client_present:
                try:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
                except OSError:
                    client_present = False
                    print("    [info] le PC Windows ne suit plus (le mouvement continue)")

        p = subprocess.Popen([EXE, ROBOT_IP, "executer", fichier],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for ligne in p.stdout:
            ligne = ligne.rstrip()
            if ligne.startswith("Q "):               # angles toutes les 40 ms : pour l'ombre, pas pour l'écran
                try:
                    transmettre({"q": [float(v) for v in ligne.split()[1:8]]})
                except ValueError:
                    pass
                continue
            print("    " + ligne)
            log.append(ligne)
            if ligne.startswith("MESURE "):
                mesures.append(json.loads(ligne[7:]))
            elif ligne.startswith("PINCE "):
                transmettre({"pince": float(ligne.split()[1])})
            transmettre({"ligne": ligne})
        code = p.wait()
        res = {"ok": code == 0, "code": code, "log": log[-30:], "mesures": mesures,
               "duree_s": round(time.time() - t0, 1)}
        with open(os.path.join(ARCHIVES, f"{stamp}_resultat.json"), "w") as f:
            json.dump({"traj": traj, "resultat": res}, f, indent=1)
        print(f"=== Fin : {'OK' if code == 0 else f'ARRET (code {code})'} en {res['duree_s']} s")
        transmettre({"fin": res})


def main():
    if not os.path.exists(EXE):
        sys.exit(f"Programme introuvable : {EXE}\n(compiler jumeau_robot.cpp, voir les instructions)")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Gestion) as srv:
        print(f"Serveur du jumeau prêt : IP Wi-Fi {ip_wifi()}, port {PORT}, robot {ROBOT_IP}"
              + ("  (confirmation sur l'Ubuntu)" if CONFIRMER else ""))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\narrêt du serveur")


if __name__ == "__main__":
    main()
