"""Lien réseau Windows -> Ubuntu (serveur_robot.py, port 5005).

    etat(ip)                      -> {"q": [7], "O_T_EE": [16], "largeur": w}
    executer(ip, traj, suivi)     -> {"ok": bool, "code": int, "log": [...], "mesures": [...]}
    serrer_crayon(ip)             -> ouvre la pince, attend le crayon, serre (début de toute mesure)
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


LARGEUR_CRAYON = 0.008      # m : largeur visée (crayon ~7-8 mm), tolérance +-8 mm
FORCE_CRAYON = 20.0         # N


def _pince(ip, etape):
    """Une seule étape de pince, sans aucun mouvement du bras (départ = pose actuelle)."""
    q = etat(ip)["q"]
    return executer(ip, {"q_depart": q, "vitesse": 0.05, "etapes": [etape],
                         "info": {"bilan_sim": "pince seule (crayon de mesure)"}})


def serrer_crayon(ip):
    """Début de toute série de mesures au crayon (27/11) : ouvrir la pince, laisser Louis glisser le
    crayon entre les doigts, serrer. Le bras ne bouge pas. Si le serveur demande une confirmation
    (--confirmer), appuyer sur Entrée dans le terminal de l'Ubuntu à chaque étape.
    Renvoie True si le crayon est serré."""
    largeur = etat(ip)["largeur"]
    if largeur < 0.012:
        r = input(f"\nLa pince est déjà fermée ({largeur * 1000:.1f} mm) : crayon déjà serré ? "
                  "Entrée = oui, garder / r = rouvrir et resserrer : ").strip().lower()
        if r != "r":
            return True
    print("\nOuverture de la pince (le bras ne bouge pas)...")
    res = _pince(ip, {"type": "ouvrir", "largeur": 0.08})
    if not res.get("ok"):
        print("[ERREUR] la pince ne s'ouvre pas : " + " | ".join(res.get("log", [])[-2:]))
        print("    Robot en lumière BLEUE (prêt) ? serveur_robot.py lancé ?")
        return False
    input(">>> Glisse le crayon entre les doigts (pointe vers le bas, ~3 cm qui dépassent), "
          "tiens-le, puis Entrée = serrer : ")
    res = _pince(ip, {"type": "saisir", "largeur": LARGEUR_CRAYON, "force": FORCE_CRAYON})
    largeur = etat(ip)["largeur"]
    if not res.get("ok") or largeur > 0.012:
        print(f"[ERREUR] crayon pas serré (pince à {largeur * 1000:.1f} mm) : "
              + " | ".join(res.get("log", [])[-2:]))
        return False
    print(f"Crayon serré (pince à {largeur * 1000:.1f} mm, {FORCE_CRAYON:.0f} N).")
    return True


if __name__ == "__main__":
    ip = sys.argv[1] if len(sys.argv) > 1 else input("IP de l'Ubuntu : ").strip()
    r = etat(ip)
    print("q =", [round(v, 4) for v in r["q"]])
    print("O_T_EE (position) =", [round(r["O_T_EE"][i], 4) for i in (12, 13, 14)], "m")
    print("largeur pince =", r["largeur"], "m")
