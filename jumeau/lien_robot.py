"""Lien réseau Windows -> Ubuntu (serveur_robot.py, port 5005).

    etat(ip)                      -> {"q": [7], "O_T_EE": [16], "largeur": w}
    executer(ip, traj, suivi)     -> {"ok": bool, "code": int, "log": [...], "mesures": [...]}
        pendant le mouvement, suivi(msg) est appelé à chaque message du robot :
        {"q": [7]} toutes les 40 ms, {"pince": w}, {"ligne": "ETAPE 3/10 ..."}

Test rapide (PowerShell) :  python jumeau/lien_robot.py 10.254.79.65
"""
import json
import socket
import sys

PORT = 5005


def _connexion(ip, msg, timeout):
    s = socket.create_connection((ip, PORT), timeout=timeout)
    s.settimeout(timeout)
    s.sendall((json.dumps(msg) + "\n").encode())
    s.shutdown(socket.SHUT_WR)
    return s


def etat(ip, timeout=6.0):
    with _connexion(ip, {"cmd": "etat"}, timeout) as s:
        rep = json.loads(s.makefile("r", encoding="utf-8").read())
    if "erreur" in rep:
        raise RuntimeError(rep["erreur"])
    return rep


def executer(ip, traj, suivi=None, timeout=900.0):
    with _connexion(ip, {"cmd": "executer", "traj": traj}, timeout) as s:
        for ligne in s.makefile("r", encoding="utf-8"):
            if not ligne.strip():
                continue
            msg = json.loads(ligne)
            if "fin" in msg:
                return msg["fin"]
            if "erreur" in msg:
                raise RuntimeError(msg["erreur"])
            if suivi is not None:
                suivi(msg)
    raise RuntimeError("liaison coupée avant la fin du mouvement")


if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else input("IP de l'Ubuntu : ").strip()
    r = etat(ip)
    print("q =", [round(v, 4) for v in r["q"]])
    print("O_T_EE (position) =", [round(r["O_T_EE"][i], 4) for i in (12, 13, 14)], "m")
    print("largeur pince =", r["largeur"], "m")
