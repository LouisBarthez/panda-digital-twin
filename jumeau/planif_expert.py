"""Expert scripté du jumeau : planifie un pick-and-place à partir des positions détectées,
le rejoue dans la scène twin (MuJoCo) et dit s'il est sûr de l'envoyer au vrai robot.

Même logique que l'expert de common/test_expert.py (au-dessus du cube -> descente -> fermeture ->
montée -> transport -> dépose -> lâcher -> remontée), mais écrite en POINTS DE PASSAGE ARTICULAIRES :
le vrai robot rejoue exactement ces angles (libfranka, MotionGenerator = mouvement articulaire
synchronisé), donc ce qui part sur le robot est ce qui a été validé dans MuJoCo.

Nouveautés par rapport à l'ancien expert :
  - la pince tourne avec le cube (lacet détecté, à 90° près) ;
  - les hauteurs suivent la table rouge inclinée (red_z) ;
  - départ de la pose RÉELLE du robot, retour à cette pose à la fin (bras hors du champ de la webcam).

Test sans webcam ni robot (PowerShell, venv activé, dans C:\\Users\\louis\\pfe_mujoco) :
    python jumeau/planif_expert.py                         -> positions de perception/detection.json
    python jumeau/planif_expert.py --cube 0.55 -0.10 30 --cible 0.60 0.15
    python jumeau/planif_expert.py --test 20               -> 20 tirages aléatoires, taux de réussite
"""
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "common"))
from twin_scene import (CUBE_HALF, HOME_Q, TARGET_H, build_twin_spec,  # noqa: E402
                        hors_du_trou, red_z)

TCP_OFFSET = 0.1034          # bride -> centre des doigts (même valeur que l'EE "Franka Hand" de Desk)
H_SURVOL = 0.15              # hauteur de survol au-dessus de la position de prise / dépose [m]
Z_PRISE = 0.005              # centre des doigts 5 mm au-dessus du centre du cube
Z_LACHER = 0.004             # le cube est lâché 4 mm au-dessus de la cible
LARGEUR_OUVERTE = 0.08
LARGEUR_CUBE = 0.04
FORCE_PRISE = 20.0           # N (cube PLA 23,3 g)
MARGE_BUTEE = 0.05           # rad, marge aux butées articulaires
DT_AFFICHAGE = 0.02          # le rejeu avance par tranches de 20 ms (une image du viewer)

# corps du robot qui ne doivent jamais toucher le décor (link0 est posé sur la table blanche)
CORPS_ROBOT = ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
               "hand", "left_finger", "right_finger"]
DOIGTS = ["left_finger", "right_finger"]


# ------------------------------------------------------------------------------- modèle
def construire_modele():
    """Scène twin + site 'tcp' au centre des doigts (le cube et la cible sont replacés ensuite)."""
    spec = build_twin_spec()
    # compensation de gravité sur le bras : le vrai Panda la fait dans son contrôleur interne et suit
    # ses consignes au dixième de mm. Sans elle, les actionneurs PD du modèle menagerie s'affaissent
    # de 5 à 7 mm sous leur propre poids (mesuré), ce qui fausserait la validation.
    for n in CORPS_ROBOT:
        spec.body(n).gravcomp = 1.0
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[1, 0, 0, 0])
    return spec.compile()


class Jumeau:
    """Modèle MuJoCo de la scène twin + outils (placer le cube, IK, rejeu)."""

    def __init__(self):
        self.model = m = construire_modele()
        self.data = mujoco.MjData(m)
        self.ik = mujoco.MjData(m)
        self.qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.dadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.f1 = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.f2 = m.jnt_qposadr[m.joint("finger_joint2").id]
        self.cube_q = m.jnt_qposadr[m.joint("cube_joint").id]
        self.cube_d = m.jnt_dofadr[m.joint("cube_joint").id]
        self.cube_b = m.body("cube").id
        self.cible_mocap = m.body_mocapid[m.body("target").id]
        self.tcp = m.site("tcp").id
        ids = [m.joint(f"joint{i}").id for i in range(1, 8)]
        self.q_lo, self.q_hi = m.jnt_range[ids, 0], m.jnt_range[ids, 1]
        self.robot_b = {m.body(n).id for n in CORPS_ROBOT}
        self.doigts_b = {m.body(n).id for n in DOIGTS}
        self.cube_geom = m.geom("cube_geom").id
        # orientation « pince verticale » de référence : celle de la posture home
        self.poser_bras(HOME_Q)
        self.R_bas = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    # ---------------------------------------------------------------- état de la scène
    def poser_bras(self, q, largeur=LARGEUR_OUVERTE):
        d = self.data
        d.qpos[self.qadr] = q
        d.qpos[self.f1] = d.qpos[self.f2] = largeur / 2
        d.qvel[:] = 0
        d.ctrl[:7] = q
        d.ctrl[7] = 255.0 if largeur > 0.05 else 0.0
        mujoco.mj_forward(self.model, d)

    def poser_objets(self, cube_xy, lacet_deg, cible_xy):
        d = self.data
        z = red_z(*cube_xy) + CUBE_HALF + 0.0005
        a = np.radians(lacet_deg)
        d.qpos[self.cube_q:self.cube_q + 7] = [cube_xy[0], cube_xy[1], z, np.cos(a / 2), 0, 0, np.sin(a / 2)]
        d.qvel[self.cube_d:self.cube_d + 6] = 0
        d.mocap_pos[self.cible_mocap] = [cible_xy[0], cible_xy[1], red_z(*cible_xy)]
        mujoco.mj_forward(self.model, d)

    def tcp_fk(self, q):
        """Position du centre des doigts pour les angles q (pour comparer au O_T_EE du vrai robot)."""
        self.ik.qpos[:] = self.data.qpos
        self.ik.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.ik)
        return self.ik.site_xpos[self.tcp].copy(), self.ik.site_xmat[self.tcp].reshape(3, 3).copy()

    # ---------------------------------------------------------------- IK
    def ik_pose(self, p_but, R_but, q_init, iters=400):
        """IK position + orientation complète (moindres carrés amortis), rappel vers HOME_Q dans
        l'espace nul. Renvoie (q, erreur_mm, erreur_deg)."""
        m, d = self.model, self.ik
        d.qpos[:] = self.data.qpos
        q = np.array(q_init, float)
        lo, hi = self.q_lo + MARGE_BUTEE, self.q_hi - MARGE_BUTEE
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.tcp]
            R = d.site_xmat[self.tcp].reshape(3, 3)
            e_p = p_but - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_but[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.tcp)
            J = np.vstack([self.jacp[:, self.dadr], self.jacr[:, self.dadr]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            # rappel vers home dans l'espace nul, SAUF pour q7 : la rotation du poignet est
            # choisie par planifier() (celle qui tourne le moins), on ne la ramène pas vers home
            rappel = 0.05 * (HOME_Q - q)
            rappel[6] = 0.0
            dq = Jp @ np.r_[e_p, e_r] + (np.eye(7) - Jp @ J) @ rappel
            n = np.abs(dq).max()
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, lo, hi)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        R = d.site_xmat[self.tcp].reshape(3, 3)
        e_p = np.linalg.norm(p_but - d.site_xpos[self.tcp]) * 1000
        e_r = np.degrees(np.arccos(np.clip((np.trace(R_but.T @ R) - 1) / 2, -1, 1)))
        return q, e_p, e_r

    def R_pince(self, rot_z):
        c, s = np.cos(rot_z), np.sin(rot_z)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.R_bas


# ------------------------------------------------------------------------------- planification
def planifier(j, q_depart, cube_xy, lacet_deg, cible_xy):
    """Points de passage de l'expert. Renvoie (etapes, message) ; etapes = None si impossible.
    Une étape : {"type": "q", "nom", "q"} | {"type": "ouvrir", "largeur"} | {"type": "saisir", "largeur", "force"}"""
    for nom, xy in (("cube", cube_xy), ("cible", cible_xy)):
        if not hors_du_trou(xy, marge=0.0):
            return None, f"{nom} trop près du passe-câbles"
    zc = red_z(*cube_xy) + CUBE_HALF                       # centre du cube
    zd = red_z(*cible_xy) + TARGET_H + CUBE_HALF + Z_LACHER  # centre du cube au moment du lâcher
    # direction de fermeture des doigts = axe y de la pince ; elle doit être perpendiculaire à deux
    # faces du cube (lacet du cube à 90° près) -> 4 orientations de pince possibles (plus une 5e
    # équivalente). On les essaie TOUTES et on garde celle qui fait le moins bouger le robot
    # (somme des plus grands déplacements articulaires entre points de passage ~ durée du mouvement) :
    # en pratique, celle qui tourne le moins le poignet depuis la pose de départ.
    phi0 = np.degrees(np.arctan2(j.R_bas[1, 1], j.R_bas[0, 1]))
    a0 = (lacet_deg - phi0 + 45) % 90 - 45
    candidats = [a0 + 90 * k for k in (-2, -1, 0, 1, 2)]
    poses = [("au-dessus du cube", [*cube_xy, zc + Z_PRISE + H_SURVOL]),
             ("prise", [*cube_xy, zc + Z_PRISE]),
             ("montee", [*cube_xy, zc + Z_PRISE + H_SURVOL]),
             ("au-dessus de la cible", [*cible_xy, zd + H_SURVOL]),
             ("depose", [*cible_xy, zd]),
             ("remontee", [*cible_xy, zd + H_SURVOL])]
    q_dep = np.array(q_depart, float)
    dernier_msg = "aucune orientation de pince possible"
    meilleur = None
    for a in candidats:
        R = j.R_pince(np.radians(a))
        qs, q = [], q_dep.copy()
        ok = True
        for nom, p in poses:
            q, ep, er = j.ik_pose(np.array(p), R, q)
            if ep > 1.0 or er > 0.5:
                ok = False
                dernier_msg = f"IK impossible pour « {nom} » (rotation pince {a:+.0f}°) : {ep:.1f} mm / {er:.1f}°"
                break
            qs.append((nom, q.copy()))
        if not ok:
            continue
        chemin = [q_dep] + [q for _, q in qs] + [q_dep]
        cout = sum(np.abs(b - c).max() for b, c in zip(chemin[:-1], chemin[1:]))
        if meilleur is None or cout < meilleur[0]:
            meilleur = (cout, a, dict(qs))
    if meilleur is not None:
        cout, a, Q = meilleur
        etapes = [{"type": "ouvrir", "largeur": LARGEUR_OUVERTE},
                  {"type": "q", "nom": "au-dessus du cube", "q": Q["au-dessus du cube"]},
                  {"type": "q", "nom": "prise", "q": Q["prise"]},
                  {"type": "saisir", "largeur": LARGEUR_CUBE, "force": FORCE_PRISE},
                  {"type": "q", "nom": "montee", "q": Q["montee"]},
                  {"type": "q", "nom": "au-dessus de la cible", "q": Q["au-dessus de la cible"]},
                  {"type": "q", "nom": "depose", "q": Q["depose"]},
                  {"type": "ouvrir", "largeur": LARGEUR_OUVERTE},
                  {"type": "q", "nom": "remontee", "q": Q["remontee"]},
                  {"type": "q", "nom": "retour", "q": q_dep.copy()}]
        dq7 = abs(Q["prise"][6] - q_dep[6])
        return etapes, f"plan OK (pince {a:+.0f}° / home, poignet tourné de {np.degrees(dq7):.0f}°)"
    return None, dernier_msg


# ------------------------------------------------------------------------------- rejeu dans MuJoCo
def _profil(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5       # quintique : vitesse et accélération nulles aux bouts


def _duree(dq):
    return float(np.abs(dq).max()) / 0.6 + 0.6           # même allure partout (~0,6 rad/s au plus)


def _contacts_interdits(j):
    """Contacts robot / décor (tables, bague) ou main / cube : renvoie la liste des paires."""
    m, d = j.model, j.data
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for br, g_autre, b_autre in ((b1, c.geom2, b2), (b2, c.geom1, b1)):
            if br not in j.robot_b or b_autre in j.robot_b or b_autre == 0 and m.geom(g_autre).name == "floor":
                continue
            if g_autre == j.cube_geom and br in j.doigts_b:
                continue                                  # les doigts ont le droit de toucher le cube
            if c.dist > -0.0005 and g_autre != j.cube_geom:
                continue                                  # effleurement (< 0,5 mm) toléré
            out.append(f"{m.body(br).name} / {m.geom(g_autre).name or m.body(b_autre).name}")
    return out


def rejouer(j, q_depart, cube_xy, lacet_deg, cible_xy, etapes):
    """Générateur : rejoue le plan dans MuJoCo par tranches de 20 ms (un `next()` = une image).
    À la fin, renvoie (StopIteration.value) un dict de bilan : ok, message, mesures."""
    m, d = j.model, j.data
    n_sub = int(round(DT_AFFICHAGE / m.opt.timestep))
    j.poser_bras(q_depart)
    j.poser_objets(cube_xy, lacet_deg, cible_xy)
    bilan = {"ok": False, "message": "", "contacts": [], "largeur_prise": None}

    def avancer(duree, consigne=None):
        n = max(1, int(round(duree / DT_AFFICHAGE)))
        for k in range(n):
            if consigne is not None:
                d.ctrl[:7] = consigne((k + 1) / n)
            mujoco.mj_step(m, d, nstep=n_sub)
            for pb in _contacts_interdits(j):
                if pb not in bilan["contacts"]:
                    bilan["contacts"].append(pb)
            yield

    yield from avancer(0.3)                               # le cube se pose
    z_cube0 = d.xpos[j.cube_b][2]
    q_cmd = np.array(q_depart, float)
    for e in etapes:
        if e["type"] == "q":
            q0, q1 = q_cmd.copy(), np.array(e["q"])
            yield from avancer(_duree(q1 - q0), lambda s, q0=q0, q1=q1: q0 + _profil(s) * (q1 - q0))
            yield from avancer(0.2)
            q_cmd = q1
        elif e["type"] == "ouvrir":
            d.ctrl[7] = 255.0
            yield from avancer(1.0)
        elif e["type"] == "saisir":
            d.ctrl[7] = 0.0
            yield from avancer(1.2)
            bilan["largeur_prise"] = float(d.qpos[j.f1] + d.qpos[j.f2])
        if bilan["contacts"]:
            bilan["message"] = "COLLISION : " + ", ".join(bilan["contacts"][:3])
            return bilan
        if e.get("nom") == "montee" and d.xpos[j.cube_b][2] - z_cube0 < 0.10:
            bilan["message"] = f"le cube n'est pas monté avec la pince (largeur {bilan['largeur_prise'] * 1000:.0f} mm)"
            return bilan
    # bilan final
    pos = d.xpos[j.cube_b]
    err = np.hypot(pos[0] - cible_xy[0], pos[1] - cible_xy[1])
    zr = pos[2] - (red_z(*cible_xy) + TARGET_H + CUBE_HALF)
    Rz = d.xmat[j.cube_b].reshape(3, 3)[:, 2]
    incl = np.degrees(np.arccos(np.clip(abs(Rz[2]), -1, 1)))
    bilan.update(erreur_depose_mm=round(err * 1000, 1), inclinaison_deg=round(incl, 1))
    if err > 0.015:
        bilan["message"] = f"cube à {err * 1000:.0f} mm de la cible"
    elif abs(zr) > 0.008 or incl > 10:
        bilan["message"] = f"cube mal posé (hauteur {zr * 1000:+.0f} mm, inclinaison {incl:.0f}°)"
    else:
        bilan["ok"] = True
        bilan["message"] = (f"OK : cube à {err * 1000:.1f} mm de la cible, "
                            f"prise à {bilan['largeur_prise'] * 1000:.1f} mm")
    return bilan


def rejouer_tout(j, *args):
    g = rejouer(j, *args)
    while True:
        try:
            next(g)
        except StopIteration as fin:
            return fin.value


# ------------------------------------------------------------------------------- format d'échange
def vers_json(etapes, q_depart, vitesse, info):
    out = []
    for e in etapes:
        e2 = dict(e)
        if "q" in e2:
            e2["q"] = [round(float(v), 5) for v in e2["q"]]
        out.append(e2)
    return {"q_depart": [round(float(v), 5) for v in q_depart], "vitesse": vitesse,
            "etapes": out, "info": info}


# ------------------------------------------------------------------------------- tests hors ligne
def _arg(nom, n):
    i = sys.argv.index(nom)
    return [float(v) for v in sys.argv[i + 1:i + 1 + n]]


def main():
    j = Jumeau()
    if "--test" in sys.argv:
        n = int(sys.argv[sys.argv.index("--test") + 1])
        rng = np.random.default_rng(0)
        ok = 0
        for k in range(n):
            while True:
                c = rng.uniform([0.40, -0.25], [0.68, 0.25])
                t = rng.uniform([0.40, -0.25], [0.68, 0.25])
                if np.linalg.norm(c - t) > 0.10 and hors_du_trou(c) and hors_du_trou(t):
                    break
            lac = rng.uniform(0, 90)
            et, msg = planifier(j, HOME_Q, c, lac, t)
            if et is None:
                print(f"{k:2d} cube ({c[0]:.2f};{c[1]:+.2f}) {lac:4.0f}° -> PLAN IMPOSSIBLE : {msg}")
                continue
            b = rejouer_tout(j, HOME_Q, c, lac, t, et)
            ok += b["ok"]
            print(f"{k:2d} cube ({c[0]:.2f};{c[1]:+.2f}) {lac:4.0f}°  cible ({t[0]:.2f};{t[1]:+.2f})"
                  f"  {msg} -> {b['message']}")
        print(f"\n{ok}/{n} réussis en simulation")
        return
    if "--cube" in sys.argv:
        cx, cy, lac = _arg("--cube", 3)
        tx, ty = _arg("--cible", 2)
    else:
        with open(os.path.join(ROOT, "perception", "detection.json")) as f:
            det = json.load(f)
        cx, cy = det["cube"]["xyz"][:2]
        lac = det["cube"].get("lacet_deg", 0.0)
        tx, ty = det["cible"]["xyz"][:2]
    et, msg = planifier(j, HOME_Q, (cx, cy), lac, (tx, ty))
    print(msg)
    if et:
        print(rejouer_tout(j, HOME_Q, (cx, cy), lac, (tx, ty), et))


if __name__ == "__main__":
    main()
