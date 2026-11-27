"""Jumeau avec la planche : la scène détectée (cube, cible, planche) -> décision -> plan -> MuJoCo -> robot.

Branche le travail du 24-26/11 (planche, politique de décision IL, générateur, détection de la planche)
sur le jumeau en direct (jumeau_live.py --planche) :

  1. la planche détectée (perception/detection_planche.py) est placée dans la scène twin ;
  2. la POLITIQUE choisit (orientation de pince, route d'approche, route de transport) :
       "il"     : le réseau de l'IL (il/decision_il.pt) classe les 180 choix, et le générateur
                  teste les choix dans le jumeau dans cet ordre et garde le 1er sûr (mode « jumeau »,
                  92 % en simulation, §7.19) ; si aucun des 10 premiers n'est sûr, le planificateur
                  complet prend le relais (le message le dit) ;
       "expert" : le planificateur complet (IK + contournements + RRT, 97 % en simulation) ;
  3. le plan est complété par un RETOUR à la pose de départ (bras hors du champ de la webcam pour la
     mesure finale), chemin libre lui aussi ;
  4. il est converti en points de passage articulaires (même format que planif_expert.planifier) :
     E le rejoue dans MuJoCo (planche comprise : tout contact = refus), R l'envoie au vrai robot.

Test hors ligne (PowerShell, venv activé, à la racine du dépôt) :
    python jumeau/planif_planche.py                       -> scène de perception/detection_planche.json
    python jumeau/planif_planche.py --test 10             -> 10 scènes tirées au hasard
    options : --politique il|expert
"""
import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "common"))
sys.path.insert(0, os.path.join(ROOT, "avec_obstacle", "common"))
sys.path.insert(0, os.path.join(ROOT, "avec_obstacle", "il"))
from planif_expert import CORPS_ROBOT, HOME_Q, TCP_OFFSET, Jumeau, rejouer_tout  # noqa: E402
from planche import ajouter_planche, pose_planche, tirer_scene  # noqa: E402
from twin_scene import CUBE_HALF, build_twin_spec, red_z  # noqa: E402

POLITIQUES = ("il", "expert")
# pose « perchée » d'observation : pince vers le bas, D435 à ~85 cm au-dessus de la table, qui voit
# toute la zone de travail ET le tag 0 (recalage), sans le boîtier de commande Franka (la 1re pose
# le voyait et le prenait pour la planche). Pose choisie par Louis le 26/11 à 17 h (caméra à 0,80 m).
OBS_Q = np.array([-0.107, 0.013, 0.007, -0.868, -0.012, 0.918, 0.72])


# ------------------------------------------------------------------------------- modèle du jumeau
def construire_modele_planche():
    """Scène du jumeau (planif_expert) + planche + boîtier D435 solide + pince et cube réglés comme
    dans twin_env (sinon le cube glisse dans la pince pendant les contournements)."""
    spec = build_twin_spec()
    for n in CORPS_ROBOT:
        spec.body(n).gravcomp = 1.0
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[1, 0, 0, 0])
    g = spec.geom("d435_boitier")
    g.contype, g.conaffinity = 1, 1
    act = spec.actuator("actuator8")
    gp, bp = list(act.gainprm), list(act.biasprm)
    gp[0], bp[0] = 14.0 / 255.0, -10.0
    act.gainprm, act.biasprm = gp, bp
    cg = spec.geom("cube_geom")
    cg.solref = [0.004, 1.0]
    cg.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    ajouter_planche(spec)
    return spec.compile()


class JumeauPlanche(Jumeau):
    """Jumeau de planif_expert avec la planche (corps mocap, fixe). Tout le reste est identique :
    poser_bras, poser_objets, tcp_fk, rejouer (qui refuse tout contact du bras avec le décor,
    planche comprise)."""

    def __init__(self):
        import planif_expert
        ancien = planif_expert.construire_modele
        planif_expert.construire_modele = construire_modele_planche
        try:
            super().__init__()
        finally:
            planif_expert.construire_modele = ancien
        m = self.model
        self.pl_mocap = m.body_mocapid[m.body("planche").id]
        self.planche = None
        self.poser_planche(None)

    def poser_planche(self, planche):
        """planche = {"posture", "x", "y", "lacet" (rad)} ou None (rangée sous la table)."""
        d = self.data
        self.planche = planche
        if planche is None:
            d.mocap_pos[self.pl_mocap] = [0.6, 0.0, -0.5]
        else:
            pq = pose_planche(planche["posture"], planche["x"], planche["y"], planche["lacet"])
            d.mocap_pos[self.pl_mocap], d.mocap_quat[self.pl_mocap] = pq[:3], pq[3:]
        mujoco.mj_forward(self.model, d)


# ------------------------------------------------------------------------------- politique
_CACHE = {}


def _generateur():
    if "gen" not in _CACHE:
        from decisions_planche import Generateur
        _CACHE["gen"] = Generateur()
    return _CACHE["gen"]


def _politique_il():
    """Classement des 180 choix par le réseau de l'IL (il/decision_il.pt)."""
    if "il" not in _CACHE:
        cwd, argv = os.getcwd(), list(sys.argv)
        sys.argv = [argv[0]]
        try:
            import train_decision_il as T            # (ce module change de dossier à l'import)
            _CACHE["il"] = T.fonction(T.charger("decision_il"), sans=False)
        finally:
            os.chdir(cwd)
            sys.argv = argv
    return _CACHE["il"]


def prechauffer():
    """Charge le générateur et le réseau (appelé une fois dans le processus de planification)."""
    _generateur()
    _politique_il()
    return True


def scene_det(cube_xy, lacet_deg, cible_xy, planche):
    """Dictionnaire « détection » au format de TwinPlancheEnv.detection() (entrée du générateur)."""
    pq = pose_planche(planche["posture"], planche["x"], planche["y"], planche["lacet"])
    return {"cube": np.array([cube_xy[0], cube_xy[1], red_z(*cube_xy) + CUBE_HALF]),
            "cube_lacet": float(np.radians(lacet_deg)),
            "cible": np.array([cible_xy[0], cible_xy[1], red_z(*cible_xy)]),
            "planche_centre": pq[:3].copy(), "planche_lacet": float(planche["lacet"]),
            "posture": planche["posture"]}


def poignet_le_moins_tourne(gen, classement, det, q0):
    """L'orientation de pince k (lacet du cube à 90° près : 5 candidats) ne change rien au trajet,
    seulement à la rotation du poignet. Le réseau l'a apprise avec des départs autour de home ; depuis
    une autre pose (pose perchée : poignet tourné de ~85° par rapport à home), son k fait faire au
    poignet un demi-tour inutile (vu par Louis le 26/11, même défaut que le 23/11 sur l'expert).
    -> on garde l'ORDRE DES ROUTES du réseau, et pour chaque paire de routes on essaie les k du plus
    petit au plus grand tour de poignet depuis la vraie pose de départ."""
    pl = gen.pl
    pl._poser_bras(np.asarray(q0, float), 0.08)
    mujoco.mj_kinematics(pl.model, pl.data)
    R0 = pl.data.site_xmat[pl.ix.tcp].reshape(3, 3)
    M = R0 @ pl.R_bas.T                                   # rotation actuelle de la pince autour de z
    b = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    angles = gen.angles_pince(det)
    ordre_k = sorted(range(len(angles)), key=lambda k: abs(angles[k] - b))
    routes = []
    for k, ra, rt in np.atleast_2d(classement):
        if (int(ra), int(rt)) not in routes:
            routes.append((int(ra), int(rt)))
    return np.array([(k, ra, rt) for ra, rt in routes for k in ordre_k])


N_PLANS_COMPARES = 3        # plans sûrs comparés avant de choisir (27/11)
MAX_ESSAIS_CHOIX = 40       # choix du classement essayés au plus (~3 s)
POIGNET_MAX_DEG = 150.0     # au-delà, tour de poignet jugé dangereux pour le câble de la D435


def _cout_plan(plan, q0):
    """(coût, tour de poignet max en degrés) : chemin articulaire total aller + retour, poignet (q7)
    compté 3 fois (c'est lui qui enroule le câble de la D435)."""
    Q = [np.asarray(q0, float)]
    for st in plan:
        if "q" in st:
            Q.extend(np.atleast_2d(np.asarray(st["q"], float)))
    Q.append(np.asarray(q0, float))
    Q = np.vstack(Q)
    dq = np.abs(np.diff(Q, axis=0)).sum(axis=0)
    poignet = float(np.degrees(np.abs(Q[:, 6] - Q[0, 6]).max()))
    return float(dq.sum() + 2 * dq[6]), poignet


def choisir_moins_tourne(gen, classement, det, q0):
    """Remplace decider() pour le vrai robot (27/11). decider() prenait le 1er choix sûr : le
    27/11 à 10 h 30 (planche debout), c'était une orientation de pince symétrique de la bonne
    (même prise, à 180°) qui n'est atteignable qu'en retournant tout le bras : poignet tourné de
    199°, câble de la D435 tiré. On garde l'ordre du réseau, mais on compare les N_PLANS_COMPARES
    premiers plans sûrs et on prend celui qui bouge le moins (poignet surtout). Ici : poignet 70°,
    chemin articulaire 15 rad au lieu de 27.
    Renvoie (plan, message, choix, rang, tour de poignet en degrés)."""
    from decisions_planche import ROUTES
    classement = [tuple(int(v) for v in c) for c in np.atleast_2d(classement)]
    cache, trouves = {}, []
    for rang, (k, ra, rt) in enumerate(classement):
        bons = [t for t in trouves if t[4] <= POIGNET_MAX_DEG]
        if len(bons) >= N_PLANS_COMPARES or (bons and rang >= MAX_ESSAIS_CHOIX):
            break               # sinon (rien ou seulement des demi-tours) on continue tout le classement
        if k not in cache:
            gen.pl.poser_scene(det)
            gen.pl.porte = None
            cache[k] = gen.poses_cles(det, q0, k)
        if cache[k] is None:
            continue
        gen.pl.poser_scene(det)
        gen.pl.porte = None
        plan, msg = gen.plan_depuis_choix(det, q0, k, ra, rt, verifier=True, pc=cache[k])
        if plan is None:
            continue
        cout, poignet = _cout_plan(plan, q0)
        trouves.append((cout, rang, (k, ra, rt), plan, poignet))
    bons = [t for t in trouves if t[4] <= POIGNET_MAX_DEG]
    if not bons:
        # rien de sûr, ou seulement des plans qui retournent le poignet : le planificateur complet
        # prend le relais (planifier_planche)
        detail = (f"seulement des demi-tours de poignet ({min(t[4] for t in trouves):.0f} deg mini)"
                  if trouves else "rien de sûr")
        return None, f"aucun choix sûr dans le jumeau : {detail}", classement[0], 0, None
    cout, rang, choix, plan, poignet = min(bons, key=lambda t: t[0])
    k, ra, rt = choix
    msg = f"k={k}, approche {ROUTES[ra]}, transport {ROUTES[rt]} ({len(trouves)} plans sûrs comparés)"
    return plan, msg, choix, rang, poignet


def planifier_planche(q_depart, cube_xy, lacet_deg, cible_xy, planche, politique="il"):
    """Renvoie (etapes, message, info) ; etapes au format de planif_expert.planifier (None si
    impossible) ; info = choix retenu, rang, durée."""
    from decisions_planche import caracteristiques
    from expert_planche import LARGEUR_OUVERTE, vers_etapes
    t0 = time.time()
    det = scene_det(cube_xy, lacet_deg, cible_xy, planche)
    q0 = np.array(q_depart, float)
    gen = _generateur()
    pl = gen.pl
    pl.poser_scene(det)
    pl.porte = None
    if not pl.libre(q0, LARGEUR_OUVERTE):
        return None, ("pose de départ du robot trop près de la planche, de la table ou du cube "
                      "(remonter le bras)"), {}
    info = {"politique": politique}
    if politique == "il":
        # le réseau a été entraîné avec des départs autour de la posture home : on lui donne cette
        # posture (ses décisions portent sur la scène) ; le générateur, lui, part de la VRAIE pose
        classement = _politique_il()(caracteristiques(det, HOME_Q), det, q0, None)
        classement = poignet_le_moins_tourne(gen, classement, det, q0)
        plan, msg, choix, rang, poignet = choisir_moins_tourne(gen, classement, det, q0)
        info.update(choix=[int(v) for v in choix], rang=int(rang))
        if poignet is not None:
            info["poignet_deg"] = round(poignet)
        if plan is None or "aucun choix sûr" in msg:
            # aucun des 10 premiers choix du réseau n'est sûr (cas rare : poteau qui barre l'approche,
            # 0 à 3/8 en simulation) -> le planificateur complet prend le relais, et on le dit
            info["secours"] = f"IL sans choix sûr ({msg})"
            politique = "expert"
    if politique == "expert":
        pl.rng = np.random.default_rng(0)
        plan, msg = pl.planifier(det, q0, retour=False)
    if plan is None:
        return None, msg, info
    # retour à la pose de départ : cube posé sur la cible, chemin libre (contournement si besoin)
    pl.poser_scene(det)
    pl.porte = None
    zc = red_z(*cible_xy) + 0.0015 + CUBE_HALF
    pl.cube_pose = np.r_[cible_xy[0], cible_xy[1], zc, pl.cube_pose[3:]]
    ch = pl.chemin_libre(plan[-1]["q"][-1], q0, LARGEUR_OUVERTE)
    if ch is None:
        return None, "pas de chemin libre pour revenir à la pose de départ", info
    plan.append({"type": "chemin", "nom": "retour", "q": ch, "largeur": LARGEUR_OUVERTE})
    info["poignet_deg"] = round(_cout_plan(plan, q0)[1])           # aller + retour, plan final
    etapes = vers_etapes(plan)
    info["duree_s"] = round(time.time() - t0, 1)
    info["n_points"] = sum(e["type"] == "q" for e in etapes)
    from decisions_planche import ROUTES
    if politique == "il" and not info.get("secours"):
        k, ra, rt = info["choix"]
        msg = (f"IL : approche {ROUTES[ra]}, transport {ROUTES[rt]} (choix n°{info['rang'] + 1} du réseau, "
               f"vérifié), {info['n_points']} points, {info['duree_s']} s, poignet {info['poignet_deg']} deg")
    elif info.get("secours"):
        msg = (f"IL : aucun choix sûr -> PLANIFICATEUR COMPLET : {msg}, {info['n_points']} points, "
               f"poignet {info['poignet_deg']} deg")
    else:
        msg = f"expert : {msg}, {info['n_points']} points, poignet {info['poignet_deg']} deg"
    if info["poignet_deg"] > POIGNET_MAX_DEG:
        msg += " ATTENTION POIGNET : ne pas faire R"
    return etapes, msg, info


# ------------------------------------------------------------------------------- déplacement simple
def planifier_deplacement(q_depart, q_but, cube_xy=None, lacet_deg=0.0, cible_xy=None, planche=None):
    """Chemin libre (planche et objets connus évités) de q_depart à q_but, en étapes du jumeau.
    Sert à aller en pose d'observation (touche O). Renvoie (etapes, message)."""
    from expert_planche import LARGEUR_OUVERTE, vers_etapes
    gen = _generateur()
    pl = gen.pl
    det = scene_det(cube_xy if cube_xy is not None else (0.9, -0.9), lacet_deg,
                    cible_xy if cible_xy is not None else (0.9, -0.9),
                    planche or {"posture": "plat", "x": 0.6, "y": -2.0, "lacet": 0.0})
    pl.poser_scene(det)
    pl.porte = None
    q0, q1 = np.array(q_depart, float), np.array(q_but, float)
    if not pl.libre(q1, LARGEUR_OUVERTE):
        return None, "pose d'arrivée en collision (planche ?)"
    if not pl.libre(q0, LARGEUR_OUVERTE):
        q_h = pl.degager(q0, LARGEUR_OUVERTE)
        if q_h is None:
            return None, "pose de départ coincée (dégager le bras à la main)"
        ch = [q0] + (pl.chemin_libre(q_h, q1, LARGEUR_OUVERTE) or [None])
        if ch[-1] is None:
            return None, "pas de chemin libre"
    else:
        ch = pl.chemin_libre(q0, q1, LARGEUR_OUVERTE)
        if ch is None:
            return None, "pas de chemin libre"
    et = vers_etapes([{"type": "chemin", "nom": "observation", "q": ch, "largeur": LARGEUR_OUVERTE}])
    return et, f"vers la pose d'observation : {len(et)} point(s)"


def rejouer_deplacement(j, q_depart, etapes):
    """Générateur : rejoue un déplacement dans le jumeau (contacts vérifiés, pas de cube à poser)."""
    from planif_expert import DT_AFFICHAGE, _contacts_interdits, _duree, _profil
    m, d = j.model, j.data
    n_sub = int(round(DT_AFFICHAGE / m.opt.timestep))
    j.poser_bras(q_depart)
    bilan = {"ok": False, "message": "", "contacts": [], "deplacement": True}
    q_cmd = np.array(q_depart, float)
    for e in etapes:
        if e["type"] != "q":
            continue
        q0, q1 = q_cmd.copy(), np.array(e["q"])
        n = max(1, int(round(_duree(q1 - q0) / DT_AFFICHAGE)))
        for k in range(n):
            d.ctrl[:7] = q0 + _profil((k + 1) / n) * (q1 - q0)
            mujoco.mj_step(m, d, nstep=n_sub)
            for pb in _contacts_interdits(j):
                if pb not in bilan["contacts"] and "cube" not in pb:
                    bilan["contacts"].append(pb)
            yield
        q_cmd = q1
        if bilan["contacts"]:
            bilan["message"] = "COLLISION : " + ", ".join(bilan["contacts"][:3])
            return bilan
    bilan["ok"] = True
    bilan["message"] = "déplacement sans contact"
    return bilan


# ------------------------------------------------------------------------------- tests hors ligne
def _arg(nom, defaut):
    return sys.argv[sys.argv.index(nom) + 1] if nom in sys.argv else defaut


def main():
    pol = _arg("--politique", "il")
    j = JumeauPlanche()
    if "--test" in sys.argv:
        n = int(_arg("--test", 10))
        rng = np.random.default_rng(int(_arg("--seed", 1)))
        ok = 0
        for k in range(n):
            sc = tirer_scene(rng)
            planche = {"posture": sc["posture"], "x": sc["planche_xy"][0], "y": sc["planche_xy"][1],
                       "lacet": sc["planche_lacet"]}
            j.poser_planche(planche)
            et, msg, _ = planifier_planche(HOME_Q, sc["cube_xy"], sc["cube_lacet_deg"], sc["cible_xy"], planche, pol)
            if et is None:
                print(f"{k:2d} {sc['posture']:7s} {'chemin' if sc['sur_chemin'] else 'ailleurs':8s} -> PAS DE PLAN : {msg}")
                continue
            b = rejouer_tout(j, HOME_Q, sc["cube_xy"], sc["cube_lacet_deg"], sc["cible_xy"], et)
            ok += b["ok"]
            print(f"{k:2d} {sc['posture']:7s} {'chemin' if sc['sur_chemin'] else 'ailleurs':8s} {msg}\n"
                  f"     -> jumeau : {b['message']}")
        print(f"\n{ok}/{n} réussis dans le jumeau (politique {pol})")
        return
    with open(os.path.join(ROOT, "perception", "detection_planche.json")) as f:
        sc = json.load(f)["scene"]
    planche = {"posture": sc["posture"], "x": sc["planche_xy"][0], "y": sc["planche_xy"][1],
               "lacet": sc["planche_lacet"]}
    j.poser_planche(planche)
    et, msg, _ = planifier_planche(HOME_Q, sc["cube_xy"], sc["cube_lacet_deg"], sc["cible_xy"], planche, pol)
    print(msg)
    if et:
        print(rejouer_tout(j, HOME_Q, sc["cube_xy"], sc["cube_lacet_deg"], sc["cible_xy"], et))


if __name__ == "__main__":
    main()
