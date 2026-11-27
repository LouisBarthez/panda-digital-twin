"""Décisions haut niveau pour le pick-and-place avec la planche : le réseau DÉCIDE, un générateur
classique EXÉCUTE.

Pourquoi : un réseau qui sort 7 petits déplacements d'angles toutes les 40 ms doit apprendre à la
fois l'IK, le profil de vitesse et le choix du chemin. Le BC plafonnait à 5 %, DAgger à ~40 %
(24/11). Ici la politique ne prend que 3 décisions par épisode :

    k        orientation de la pince pour la prise (lacet du cube à 90° près : 5 candidats)
    approche chemin départ -> au-dessus du cube     parmi ROUTES
    transport chemin montée -> au-dessus de la cible parmi ROUTES

    ROUTES = direct | dessus (par-dessus la planche) | cote+ / cote- (longer un grand côté)
             | bout+ / bout- (contourner un bout)

Le générateur fait le reste, comme le MotionGenerator du vrai robot : IK des points clés (au-dessus
du cube, prise, dépose ; il vérifie qu'ils sont accessibles sans collision, comme le ferait MoveIt,
avec plusieurs postures du coude), points de passage de la route choisie, trajectoire articulaire
lisse. Il ne CHERCHE PAS de chemin : si la route choisie traverse la planche, l'épisode échoue.
C'est donc bien la décision qui fait réussir ou échouer.

Le même module donne :
  - evaluer_options() : pour chaque orientation et chaque route, faisable ou non et coût (rad).
    -> étiquette de l'« expert discret » (option la moins chère parmi les faisables) pour l'IL,
       et vérité pour analyser les choix ;
  - plan_depuis_choix() : plan exécutable (même format que expert_planche) ;
  - executer() : joue un plan dans TwinPlancheEnv (même suivi que l'expert) ;
  - caracteristiques() : les entrées du réseau (scène vue par la webcam + pose de départ).
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from expert_planche import (H_APPROCHE, LARGEUR_CUBE, LARGEUR_OUVERTE, MARGE_PLANCHE,  # noqa: E402
                            Z_LACHER, Z_PRISE, Planificateur, cout, densifier)
from planche import EMPREINTE, HAUTEUR, POSTURES  # noqa: E402
from twin_env import DQ_MAX, MAX_STEPS  # noqa: E402
from twin_scene import CUBE_HALF, TARGET_H, red_z  # noqa: E402

ROUTES = ("direct", "dessus", "cote+", "cote-", "bout+", "bout-")
#   direct : ligne droite articulaire ; dessus : monter, traverser au-dessus de la planche, redescendre
#   cote+/- : longer un grand côté (2 coins) ; bout+/- : contourner un bout (2 coins)
N_K, N_R = 5, len(ROUTES)
COMBOS = np.array([(k, a, t) for k in range(N_K) for a in range(N_R) for t in range(N_R)])   # 180 choix
# Comment le générateur utilise le CLASSEMENT des 180 choix donné par la politique :
#   brut       : il exécute le 1er choix, tel quel (la politique seule) ;
#   accessible : il saute les choix dont l'orientation de pince n'a pas de prise/dépose accessible
#                (il le vérifie déjà, comme MoveIt) -> la politique garde le choix, le générateur
#                refuse seulement l'impossible ;
#   jumeau     : il teste dans le jumeau (collisions de toute la trajectoire, sans l'exécuter) les
#                choix dans l'ordre de la politique et exécute le 1er sûr (au plus MAX_ESSAIS_JUMEAU).
#                C'est l'idée du jumeau numérique (niveau 4) : vérifier avant d'agir.
MODES = ("brut", "accessible", "jumeau")
MAX_ESSAIS_JUMEAU = 10
ECART = MARGE_PLANCHE + 0.13          # distance des points de contournement au bord de la planche
N_FEAT = 23


def caracteristiques(det, q0):
    """Entrées du réseau : ce que la webcam donne (cube, cible, planche) + pose de départ du bras.
    23 valeurs, toutes en repère robot."""
    c, t, p = det["cube"], det["cible"], det["planche_centre"]
    po = det["posture"]
    a, l = det["cube_lacet"], det["planche_lacet"]
    return np.r_[c[:2], np.cos(4 * a), np.sin(4 * a), t[:2], t[:2] - c[:2],
                 p[:2], p[2] + HAUTEUR[po] / 2, np.cos(2 * l), np.sin(2 * l),
                 [float(po == x) for x in POSTURES], q0].astype(np.float32)


class Generateur:
    def __init__(self, seed=0):
        self.pl = Planificateur(seed)

    # ---------------------------------------------------------------- géométrie des routes
    def points_route(self, route, p_a, p_b, z_bas):
        """Points de passage cartésiens de la route (dans l'ordre du trajet p_a -> p_b)."""
        if route == "direct":
            return []
        det = self.pl.det
        po, (x, y), lac = det["posture"], det["planche_centre"][:2], det["planche_lacet"]
        L, l = EMPREINTE[po]
        c = np.array([x, y])
        u = np.array([np.cos(lac), np.sin(lac)])
        v = np.array([-u[1], u[0]])
        if route == "dessus":
            # monter à la verticale, traverser à hauteur, redescendre (un seul point au milieu
            # faisait plonger la trajectoire articulaire dans la planche)
            haut = det["planche_centre"][2] + HAUTEUR[po] / 2
            h = max(z_bas, haut + MARGE_PLANCHE + 0.08)
            return [np.r_[p_a[:2], max(h, p_a[2])], np.r_[p_b[:2], h]]
        signe = 1.0 if route.endswith("+") else -1.0
        if route.startswith("cote"):        # longer le grand côté v = signe
            pts = [c + su * (L / 2 + ECART) * u + signe * (l / 2 + ECART) * v for su in (-1, 1)]
        else:                               # contourner le bout u = signe
            pts = [c + signe * (L / 2 + ECART) * u + sv * (l / 2 + ECART) * v for sv in (-1, 1)]
        pts.sort(key=lambda q: np.linalg.norm(q - p_a[:2]))     # le plus proche du départ d'abord
        return [np.r_[q, z_bas] for q in pts]

    def chemin_route(self, qa, qb, p_a, p_b, largeur, R, route, z_bas, verifier):
        """Chemin articulaire de la route : IK de chaque point de passage (plusieurs postures de
        départ). verifier=True : tous les segments doivent être libres (étiquettes, analyse) ;
        verifier=False : seulement des configurations sans collision (exécution d'un choix)."""
        pl = self.pl
        pts, q_prec = [qa], qa
        for p in self.points_route(route, p_a, p_b, z_bas):
            trouve = None
            for g in (q_prec, (q_prec + qb) / 2, qb):
                qv, ok = pl.ik(p, R, g, q_ref=g)
                if not ok or not pl.libre(qv, largeur):
                    continue
                if verifier and not pl.segment_libre(q_prec, qv, largeur):
                    continue
                trouve = qv
                break
            if trouve is None:
                return None
            pts.append(trouve)
            q_prec = trouve
        if verifier and not pl.segment_libre(q_prec, qb, largeur):
            return None
        return pts + [qb]

    # ---------------------------------------------------------------- poses clés
    def _poses(self, det, q0):
        c, t = det["cube"], det["cible"]
        zc = red_z(*c[:2]) + CUBE_HALF
        zd = red_z(*t[:2]) + TARGET_H + CUBE_HALF + Z_LACHER
        P = {"dessus_cube": np.r_[c[:2], zc + Z_PRISE + H_APPROCHE], "prise": np.r_[c[:2], zc + Z_PRISE],
             "dessus_cible": np.r_[t[:2], zd + H_APPROCHE], "depose": np.r_[t[:2], zd]}
        pl = self.pl
        pl._poser_bras(q0, LARGEUR_OUVERTE)
        mujoco.mj_kinematics(pl.model, pl.data)
        P["depart"] = pl.data.site_xpos[pl.ix.tcp].copy()
        return P, zd

    def angles_pince(self, det):
        pl = self.pl
        phi0 = np.degrees(np.arctan2(pl.R_bas[1, 1], pl.R_bas[0, 1]))
        a0 = (np.degrees(det["cube_lacet"]) - phi0 + 45) % 90 - 45
        return [a0 + 90 * k for k in (-2, -1, 0, 1, 2)]

    def poses_cles(self, det, q0, k):
        """Pour l'orientation k : (R, porte, descente, depose) accessibles sans collision, ou None.
        Même logique que l'expert (plusieurs postures du coude si la naturelle touche)."""
        pl = self.pl
        # postures du coude tirées de façon reproductible : même scène + même k -> mêmes poses
        cle = np.round(np.r_[det["cube"], det["cube_lacet"], det["cible"], q0, k], 4)
        pl.rng = np.random.default_rng(abs(hash(tuple(cle.tolist()))) % 2 ** 32)
        P, zd = self._poses(det, q0)
        R = pl.R_pince(np.radians(self.angles_pince(det)[k]))
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, pl.cube_pose[3:])
        porte = (R.T @ (pl.cube_pose[:3] - P["prise"]), R.T @ Rc.reshape(3, 3))
        cube_depart = pl.cube_pose.copy()
        cube_sur_cible = np.r_[det["cible"][:2], zd - Z_LACHER, cube_depart[3:]]
        descente = None
        for g, ref in pl.graines(q0, n=10):
            q1, ok = pl.ik(P["dessus_cube"], R, g, q_ref=ref)
            if not ok:
                continue
            d = pl.ligne_cartesienne(q1, P["dessus_cube"], P["prise"], R)
            if d is None:
                continue
            pl.porte = None
            if not (all(pl.libre(q, LARGEUR_OUVERTE, "prise") for q in d) and pl.libre(d[-1], LARGEUR_CUBE, "prise")):
                continue
            pl.porte = porte
            ok_m = all(pl.libre(q, LARGEUR_CUBE, "prise") for q in d[::-1][1:])
            pl.porte = None
            if ok_m:
                descente = d
                break
        if descente is None:
            return None
        depose = None
        for g, ref in pl.graines(descente[0], n=10):
            q4, ok = pl.ik(P["dessus_cible"], R, g, q_ref=ref)
            if not ok:
                continue
            d = pl.ligne_cartesienne(q4, P["dessus_cible"], P["depose"], R)
            if d is None:
                continue
            pl.porte = porte
            ok_d = all(pl.libre(q, LARGEUR_CUBE, "prise") for q in d[:-1])
            pl.porte = None
            pl.cube_pose = cube_sur_cible
            ok_r = ok_d and all(pl.libre(q, LARGEUR_OUVERTE, "prise") for q in d[::-1][1:])
            pl.cube_pose = cube_depart
            if ok_r:
                depose = d
                break
        if depose is None:
            return None
        return {"R": R, "porte": porte, "descente": descente, "depose": depose, "P": P}

    # ---------------------------------------------------------------- toutes les options
    def evaluer_options(self, det, q0):
        """Faisabilité et coût de chaque option. Renvoie un dict :
            ok_k[k]           poses clés accessibles pour l'orientation k
            ok_app[k, r], c_app[k, r]     route d'approche r (départ -> au-dessus du cube)
            ok_tr[k, r],  c_tr[k, r]      route de transport r (cube tenu)
            c_fixe[k]         coût des segments verticaux (descente, montée, dépose, remontée)
            meilleur          (k, r_app, r_tr) le moins cher parmi les faisables, ou None"""
        pl = self.pl
        pl.poser_scene(det)
        pl.porte = None
        out = {"ok_k": np.zeros(N_K, bool), "ok_app": np.zeros((N_K, N_R), bool),
               "ok_tr": np.zeros((N_K, N_R), bool), "c_app": np.full((N_K, N_R), np.inf),
               "c_tr": np.full((N_K, N_R), np.inf), "c_fixe": np.full(N_K, np.inf)}
        if not pl.libre(q0, LARGEUR_OUVERTE):
            out["meilleur"] = None
            return out
        for k in range(N_K):
            pc = self.poses_cles(det, q0, k)
            if pc is None:
                continue
            out["ok_k"][k] = True
            P, R, d, dp = pc["P"], pc["R"], pc["descente"], pc["depose"]
            out["c_fixe"][k] = 2 * cout(d) + 2 * cout(dp)
            for r, route in enumerate(ROUTES):
                pl.porte = None
                ch = self.chemin_route(q0, d[0], P["depart"], P["dessus_cube"], LARGEUR_OUVERTE, R, route,
                                       P["dessus_cube"][2] + 0.05, verifier=True)
                if ch is not None:
                    out["ok_app"][k, r], out["c_app"][k, r] = True, cout(ch)
                pl.porte = pc["porte"]
                ch = self.chemin_route(d[0], dp[0], P["dessus_cube"], P["dessus_cible"], LARGEUR_CUBE, R, route,
                                       max(P["dessus_cube"][2], P["dessus_cible"][2]), verifier=True)
                pl.porte = None
                if ch is not None:
                    out["ok_tr"][k, r], out["c_tr"][k, r] = True, cout(ch)
        best = None
        for k in range(N_K):
            if not out["ok_k"][k] or not out["ok_app"][k].any() or not out["ok_tr"][k].any():
                continue
            ra, rt = int(np.argmin(out["c_app"][k])), int(np.argmin(out["c_tr"][k]))
            tot = out["c_app"][k, ra] + out["c_tr"][k, rt] + out["c_fixe"][k]
            if best is None or tot < best[0]:
                best = (tot, k, ra, rt)
        out["meilleur"] = None if best is None else best[1:]
        out["cout_meilleur"] = None if best is None else best[0]
        return out

    # ---------------------------------------------------------------- un choix -> un plan
    def plan_depuis_choix(self, det, q0, k, r_app, r_tr, verifier=False, pc=None):
        """Plan (format expert_planche) pour la décision (k, r_app, r_tr), SANS chercher de chemin.
        Renvoie (plan, message) ; plan None si l'orientation k n'a pas de poses clés accessibles
        ou si un point de la route n'a pas d'IK (ou, avec verifier=True, si la trajectoire touche
        quelque chose dans le jumeau). pc : poses clés déjà calculées pour k (cache)."""
        pl = self.pl
        pl.poser_scene(det)
        pl.porte = None
        if pc is None:
            pc = self.poses_cles(det, q0, k)
        if pc is None:
            return None, f"orientation {k} : prise ou dépose inaccessible"
        P, R, d, dp = pc["P"], pc["R"], pc["descente"], pc["depose"]
        app = self.chemin_route(q0, d[0], P["depart"], P["dessus_cube"], LARGEUR_OUVERTE, R, ROUTES[r_app],
                                P["dessus_cube"][2] + 0.05, verifier=verifier)
        pl.porte = pc["porte"]
        tr = self.chemin_route(d[0], dp[0], P["dessus_cube"], P["dessus_cible"], LARGEUR_CUBE, R, ROUTES[r_tr],
                               max(P["dessus_cube"][2], P["dessus_cible"][2]), verifier=verifier)
        pl.porte = None
        if app is None or tr is None:
            return None, ("trajectoire en collision dans le jumeau" if verifier else "point de passage sans IK")
        plan = [{"type": "ouvrir"},
                {"type": "chemin", "nom": "approche", "q": app, "largeur": LARGEUR_OUVERTE},
                {"type": "chemin", "nom": "descente", "q": d, "largeur": LARGEUR_OUVERTE},
                {"type": "saisir"},
                {"type": "chemin", "nom": "montee", "q": d[::-1], "largeur": LARGEUR_CUBE},
                {"type": "chemin", "nom": "transport", "q": tr, "largeur": LARGEUR_CUBE},
                {"type": "chemin", "nom": "depose", "q": dp, "largeur": LARGEUR_CUBE},
                {"type": "ouvrir"},
                {"type": "chemin", "nom": "remontee", "q": dp[::-1], "largeur": LARGEUR_OUVERTE}]
        return plan, f"k={k}, approche {ROUTES[r_app]}, transport {ROUTES[r_tr]}"


def decider(gen, classement, det, q0, mode="brut"):
    """Du classement des choix (meilleur d'abord) à un plan, selon le mode (voir MODES).
    Renvoie (plan, message, choix retenu, rang du choix dans le classement)."""
    classement = [tuple(int(v) for v in c) for c in np.atleast_2d(classement)]
    if mode == "brut":
        plan, msg = gen.plan_depuis_choix(det, q0, *classement[0])
        return plan, msg, classement[0], 0
    cache, secours = {}, None
    essais = 0
    for rang, (k, ra, rt) in enumerate(classement):
        if k not in cache:
            gen.pl.poser_scene(det)
            gen.pl.porte = None
            cache[k] = gen.poses_cles(det, q0, k)
        if cache[k] is None:
            continue                                     # orientation impossible : refusée
        if mode == "accessible":
            plan, msg = gen.plan_depuis_choix(det, q0, k, ra, rt, pc=cache[k])
            return plan, msg + (f" (choix n°{rang + 1})" if rang else ""), (k, ra, rt), rang
        if secours is None:
            secours = (k, ra, rt, rang)
        plan, msg = gen.plan_depuis_choix(det, q0, k, ra, rt, verifier=True, pc=cache[k])
        essais += 1
        if plan is not None:
            return plan, msg + (f" (choix n°{rang + 1}, vérifié dans le jumeau)"), (k, ra, rt), rang
        if essais >= MAX_ESSAIS_JUMEAU:
            break
    if secours is None:
        return None, "aucune orientation de pince accessible", classement[0], 0
    k, ra, rt, rang = secours                            # rien de sûr : on joue le 1er accessible
    plan, msg = gen.plan_depuis_choix(det, q0, k, ra, rt, pc=cache[k])
    return plan, msg + " (aucun choix sûr trouvé)", (k, ra, rt), rang


class SuiveurPlan:
    """Joue un plan pas à pas (pour la vidéo et le viewer) : action(env) à chaque pas."""

    def __init__(self, plan):
        self.refs = densifier(plan) if plan is not None else None
        self.k = 0

    def fini(self):
        return self.refs is None or self.k >= len(self.refs)

    def action(self, env):
        if self.refs is None:
            return np.r_[np.zeros(7), 1.0]
        q_ref, ouverte, _ = self.refs[min(self.k, len(self.refs) - 1)]
        a = np.clip((q_ref - env.q_cmd) / DQ_MAX, -1, 1)
        if self.k < len(self.refs) and np.abs(q_ref - env.q_cmd).max() <= DQ_MAX * 1.2:
            self.k += 1
        return np.r_[a, 1.0 if ouverte else -1.0]


def executer(env, plan, pas_stable=25):
    """Joue un plan dans l'env (même suivi que l'expert). Renvoie les mesures de l'épisode."""
    if plan is None:
        return {"succes": False, "collision": "", "pas": None, "longueur_q": None, "plan": False}
    refs = densifier(plan)
    k, stable, t_succes, info = 0, 0, None, {}
    for _ in range(MAX_STEPS):
        q_ref, ouverte, _ = refs[min(k, len(refs) - 1)]
        a = np.clip((q_ref - env.q_cmd) / DQ_MAX, -1, 1)
        if k < len(refs) and np.abs(q_ref - env.q_cmd).max() <= DQ_MAX * 1.2:
            k += 1
        _, _, term, trunc, info = env.step(np.r_[a, 1.0 if ouverte else -1.0])
        if info["is_success"]:
            stable += 1
            t_succes = t_succes or info["pas"]
        else:
            stable, t_succes = 0, None
        if term or trunc or (k >= len(refs) and stable >= pas_stable):
            break
    return {"succes": bool(info.get("is_success")), "collision": info.get("collision", ""),
            "pas": t_succes, "longueur_q": info.get("longueur_q"), "plan": True}


def recompense(res):
    """Récompense d'un épisode pour le RL : réussir vite rapporte, toucher la planche coûte cher."""
    if res["succes"]:
        return 10.0 - 0.04 * res["pas"]          # -1 par seconde
    if res["collision"]:
        return -10.0
    return -5.0                                   # plan impossible ou cube pas posé


# ------------------------------------------------------------------------------- évaluation commune
def evaluer_decisions(choisir, n=100, nom="", view=False, mode="brut"):
    """Évalue une politique de décision sur les scènes d'évaluation (les mêmes que pour l'expert et
    le BC : seeds 0..n-1 faisables pour le planificateur complet).
    choisir(feat, det, q0, env) -> classement des choix (tableau (N, 3), meilleur d'abord) ou un
    seul choix (k, r_app, r_tr). mode : voir MODES. Renvoie la liste des résultats."""
    import time
    from eval_planche import scenes_eval
    from twin_env import TwinPlancheEnv
    env = TwinPlancheEnv(render_mode="human" if view else None)
    gen = Generateur()
    res, t0 = [], time.time()
    scenes = scenes_eval(n)
    for i, (seed, sc) in enumerate(scenes[:5] if view else scenes):
        env.reset(seed=seed, options={"scene": sc["scene"], "q0": sc["q0"]})
        det, q0 = env.detection(), env.q_cmd.copy()
        classement = choisir(caracteristiques(det, q0), det, q0, env)
        plan, msg, choix, rang = decider(gen, classement, det, q0, mode)
        r = executer(env, plan)
        r.update(seed=seed, posture=sc["scene"]["posture"], sur_chemin=sc["scene"]["sur_chemin"],
                 choix=[int(v) for v in choix], rang=int(rang), message=msg)
        res.append(r)
        if view:
            print(f"  scène {seed} ({r['posture']}) : {msg} -> {'REUSSI' if r['succes'] else 'ECHEC ' + r['collision']}")
        elif (i + 1) % 20 == 0:
            print(f"  {nom} {i + 1}/{len(scenes)} scènes : {sum(x['succes'] for x in res)} réussies "
                  f"({time.time() - t0:.0f} s)", flush=True)
    env.close()
    return res
