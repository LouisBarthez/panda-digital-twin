"""Expert du pick-and-place AVEC la planche : un planificateur de trajectoire.

L'ancien expert (points de passage fixes, survol à 15 cm) ne sait pas éviter un obstacle quelconque.
Celui-ci cherche, pour chaque scène, le chemin articulaire le plus court SANS COLLISION :

  1. points clés par IK (position + orientation de la pince), pour chacune des orientations de pince
     compatibles avec le lacet du cube (à 90° près) : au-dessus du cube (6 cm), prise, montée,
     au-dessus de la cible, dépose, remontée ;
  2. segments verticaux (descente, montée, dépose) : lignes droites cartésiennes ;
  3. segments libres (départ -> au-dessus du cube, montée -> au-dessus de la cible, [retour]) :
     ligne droite articulaire si elle est libre ; sinon, candidats = passages par des points de
     contournement autour de la planche (par chaque côté, par-dessus) + RRT-Connect ; chaque chemin
     est raccourci et on garde le plus court ;
  4. on garde l'orientation de pince qui donne le chemin total le plus court.

Le test de collision porte sur TOUT le bras (+ boîtier D435, + cube quand il est tenu), avec des
marges : 2 cm à la planche (planche gonflée de 2 cm dans le modèle du planificateur), 1 cm aux tables. Il tourne sur une copie du modèle (le planificateur ne
touche jamais à la simulation).

« Le plus court » = somme, sur le chemin, du plus grand déplacement articulaire entre deux points
(~ durée d'un mouvement articulaire synchronisé, comme le MotionGenerator de libfranka).

Utilisation :
    from expert_planche import ExpertPlanche
    expert = ExpertPlanche()
    obs, _ = env.reset(seed=0)
    expert.reset(env)                 # planifie à partir de env.detection() (avec le biais de détection)
    action = expert.action(env)       # à chaque pas

    python avec_obstacle/common/expert_planche.py --test 30          -> 30 scènes, taux de réussite, temps de calcul
    python avec_obstacle/common/expert_planche.py --view 5           -> 5 scènes dans le viewer
"""
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from twin_env import (DQ_MAX, Indices, TwinPlancheEnv, construire_spec)  # noqa: E402
from twin_scene import CUBE_HALF, HOME_Q, TARGET_H, red_z  # noqa: E402
from planche import pose_planche  # noqa: E402

H_APPROCHE = 0.06            # hauteur des points « au-dessus » (prise / dépose) [m]
Z_PRISE = 0.005              # centre des doigts 5 mm au-dessus du centre du cube
Z_LACHER = 0.004             # cube lâché 4 mm au-dessus de la cible
MARGE_PLANCHE = 0.02         # distance mini bras / cube tenu  <->  planche
MARGE_TABLE = 0.01           # distance mini bras / cube tenu  <->  tables (segments libres)
MARGE_CUBE = 0.005           # distance mini bras  <->  cube posé (segments libres)
MARGE_BUTEE = 0.05           # rad
PAS_TEST = 0.01              # rad : finesse du test de collision le long d'un segment
PAS_RRT = 0.2                # rad : longueur max d'une branche du RRT
ITER_RRT = 4000
ESSAIS_RRT = 4               # RRT relancé plusieurs fois, on garde le chemin le plus court
DQ_EXPERT = 0.045            # rad par pas au plus (< DQ_MAX) : vitesse de pointe de l'expert
PAS_FERMETURE, PAS_OUVERTURE = 20, 15     # pas d'attente (40 ms) pour la pince
LARGEUR_OUVERTE, LARGEUR_CUBE = 0.08, 0.04


def cout(chemin):
    """Durée approchée d'un chemin articulaire : somme des plus grands déplacements par segment."""
    return float(sum(np.abs(b - a).max() for a, b in zip(chemin[:-1], chemin[1:])))


class Planificateur:
    def __init__(self, seed=0):
        # copie du modèle pour le planificateur, avec la planche GONFLÉE de MARGE_PLANCHE de chaque
        # côté (on teste ensuite l'interpénétration). Une simple marge de contact ne suffit pas :
        # MuJoCo ne génère pas toujours le contact à distance non nulle (plan validé à 6 mm de la
        # planche, vu le 24/11). Et il faut gonfler AVANT la compilation : changer geom_size après
        # coup ne met pas à jour l'arbre de tri des contacts (BVH), la planche devient invisible.
        spec = construire_spec()
        g_pl = spec.geom("planche_geom")
        g_pl.size = [v + MARGE_PLANCHE for v in g_pl.size]
        noms_tables = (["table_blanche"] + [f"table_rouge_{c}" for c in "abcd"]
                       + [f"table_rouge_coin{i}" for i in range(4)] + [f"bague{k}" for k in range(16)])
        for nom in noms_tables:                        # marges de contact, elles aussi avant compilation
            spec.geom(nom).margin = MARGE_TABLE
        spec.geom("cube_geom").margin = MARGE_CUBE
        self.model = m = spec.compile()
        self.data = mujoco.MjData(m)
        self.ix = ix = Indices(m)
        self.rng = np.random.default_rng(seed)
        self.lo, self.hi = ix.q_lo + MARGE_BUTEE, ix.q_hi - MARGE_BUTEE
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))
        self.porte = None            # pose du cube dans le repère du tcp quand il est tenu
        self.n_tests = 0
        self.d_ik = mujoco.MjData(m)
        self._poser_bras(HOME_Q, LARGEUR_OUVERTE)
        mujoco.mj_kinematics(m, self.data)
        self.R_bas = self.data.site_xmat[ix.tcp].reshape(3, 3).copy()

    # ---------------------------------------------------------------- scène
    def poser_scene(self, det):
        """det = env.detection() : cube, lacet, cible, planche (ce que voit la webcam)."""
        m, d, ix = self.model, self.data, self.ix
        c = det["cube"]
        a = det["cube_lacet"]
        self.cube_pose = np.r_[c[:2], red_z(*c[:2]) + CUBE_HALF, np.cos(a / 2), 0, 0, np.sin(a / 2)]
        d.qpos[ix.cube_q:ix.cube_q + 7] = self.cube_pose
        d.mocap_pos[ix.cible_mocap] = det["cible"]
        pq = pose_planche(det["posture"], *det["planche_centre"][:2], det["planche_lacet"])
        d.mocap_pos[ix.pl_mocap], d.mocap_quat[ix.pl_mocap] = pq[:3], pq[3:]
        self.det = det

    def _poser_bras(self, q, largeur):
        d, ix = self.data, self.ix
        d.qpos[ix.qadr] = q
        d.qpos[ix.f1] = d.qpos[ix.f2] = largeur / 2

    # ---------------------------------------------------------------- collisions
    def libre(self, q, largeur, phase="libre"):
        """True si la configuration q est sans collision.
        phase : "libre" (marges complètes) ou "prise" (segments verticaux près du cube : doigts autorisés
        à toucher le cube, bras et cube tenu autorisés à frôler la table).
        Si self.porte est défini, le cube est tenu : il suit la pince et compte comme une partie du bras."""
        m, d, ix = self.model, self.data, self.ix
        self._poser_bras(q, largeur)
        if self.porte is not None:
            mujoco.mj_kinematics(m, d)
            p, R = d.site_xpos[ix.tcp], d.site_xmat[ix.tcp].reshape(3, 3)
            pc = p + R @ self.porte[0]
            Rc = R @ self.porte[1]
            qc = np.zeros(4)
            mujoco.mju_mat2Quat(qc, Rc.flatten())
            d.qpos[ix.cube_q:ix.cube_q + 7] = np.r_[pc, qc]
        else:
            d.qpos[ix.cube_q:ix.cube_q + 7] = self.cube_pose
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        self.n_tests += 1
        tenu = self.porte is not None
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            r1, r2 = b1 in ix.robot_b, b2 in ix.robot_b
            cube1, cube2 = g1 == ix.cube_g, g2 == ix.cube_g
            if not (r1 or r2 or ((cube1 or cube2) and tenu)):
                continue                                   # décor / décor, cube posé / décor
            autre = g2 if (r1 or (cube1 and tenu)) else g1
            moi_b = b1 if autre == g2 else b2
            moi_cube = (g1 if autre == g2 else g2) == ix.cube_g
            if autre == ix.pl_g:
                if c.dist < 0:                             # planche gonflée de MARGE_PLANCHE
                    return False
            elif autre in ix.table_g:
                lim = MARGE_TABLE if phase == "libre" else (-0.0005 if not moi_cube else -0.002)
                if c.dist < lim:
                    return False
            elif autre == ix.cube_g:
                if tenu:
                    continue                               # cube tenu / doigts : normal
                if phase == "prise" and moi_b in ix.doigts_b:
                    continue
                if c.dist < (MARGE_CUBE if phase == "libre" else 0.0):
                    return False
            elif m.geom_bodyid[autre] in ix.robot_b:
                if c.dist < 0 and not (moi_cube and m.geom_bodyid[autre] in ix.doigts_b):
                    return False                           # auto-collision du bras
            # (sol : jamais atteint)
        return True

    def segment_libre(self, qa, qb, largeur, phase="libre"):
        n = int(np.ceil(np.abs(qb - qa).max() / PAS_TEST))
        for k in range(1, n + 1):
            if not self.libre(qa + (qb - qa) * k / n, largeur, phase):
                return False
        return True

    # ---------------------------------------------------------------- IK
    def ik(self, p_but, R_but, q_init, iters=300, q_ref=None):
        """IK position + orientation (moindres carrés amortis). Dans l'espace nul, rappel vers q_ref
        (HOME_Q par défaut) : changer q_ref change la posture du coude pour la même pose de pince."""
        m, d, ix = self.model, self.d_ik, self.ix
        q_ref = HOME_Q if q_ref is None else np.asarray(q_ref)
        d.qpos[:] = self.data.qpos
        q = np.array(q_init, float)
        for _ in range(iters):
            d.qpos[ix.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[ix.tcp]
            R = d.site_xmat[ix.tcp].reshape(3, 3)
            e_p = p_but - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_but[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 1e-3:
                break
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, ix.tcp)
            J = np.vstack([self.jacp[:, ix.dadr], self.jacr[:, ix.dadr]])
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
            rappel = 0.05 * (q_ref - q)
            rappel[6] = 0.0
            dq = Jp @ np.r_[e_p, e_r] + (np.eye(7) - Jp @ J) @ rappel
            nmax = np.abs(dq).max()
            if nmax > 0.2:
                dq *= 0.2 / nmax
            q = np.clip(q + dq, self.lo, self.hi)
        d.qpos[ix.qadr] = q
        mujoco.mj_kinematics(m, d)
        ok = (np.linalg.norm(p_but - d.site_xpos[ix.tcp]) < 1e-3 and
              np.linalg.norm(0.5 * sum(np.cross(d.site_xmat[ix.tcp].reshape(3, 3)[:, i], R_but[:, i])
                                       for i in range(3))) < 0.01)
        return q, ok

    def R_pince(self, rot_z):
        c, s = np.cos(rot_z), np.sin(rot_z)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self.R_bas

    def graines(self, q_init, n=10):
        """Configurations de départ pour l'IK : q_init d'abord, puis des postures tirées au hasard
        (autres positions du coude)."""
        yield np.asarray(q_init, float), None
        for _ in range(n):
            g = np.clip(np.asarray(q_init) + self.rng.normal(0, 0.8, 7), self.lo, self.hi)
            yield g, g

    def ligne_cartesienne(self, q_a, p_a, p_b, R, n=4):
        """Points articulaires d'une ligne droite cartésienne p_a -> p_b (pince d'orientation R)."""
        qs, q = [q_a], q_a
        for k in range(1, n + 1):
            q, ok = self.ik(p_a + (p_b - p_a) * k / n, R, q, q_ref=q)      # garder la posture du coude
            if not ok:
                return None
            qs.append(q)
        return qs

    # ---------------------------------------------------------------- RRT-Connect
    def _etendre(self, arbre, parents, q_vers, largeur, phase):
        dist = [np.abs(q_vers - q).max() for q in arbre]
        i = int(np.argmin(dist))
        q_pres = arbre[i]
        delta = q_vers - q_pres
        n = np.abs(delta).max()
        q_new = q_vers if n <= PAS_RRT else q_pres + delta * PAS_RRT / n
        if self.segment_libre(q_pres, q_new, largeur, phase):
            arbre.append(q_new)
            parents.append(i)
            return len(arbre) - 1, np.abs(q_new - q_vers).max() < 1e-9
        return None, False

    def rrt_connect(self, qa, qb, largeur, phase):
        A, pa = [qa], [-1]
        B, pb = [qb], [-1]
        for it in range(ITER_RRT):
            q_rand = self.rng.uniform(self.lo, self.hi) if self.rng.random() > 0.1 else B[0]
            i, _ = self._etendre(A, pa, q_rand, largeur, phase)
            if i is not None:
                # l'autre arbre essaie de rejoindre le nouveau point (glouton)
                while True:
                    j, atteint = self._etendre(B, pb, A[i], largeur, phase)
                    if j is None:
                        break
                    if atteint:
                        ca, k = [], i
                        while k >= 0:
                            ca.append(A[k])
                            k = pa[k]
                        cb, k = [], j
                        while k >= 0:
                            cb.append(B[k])
                            k = pb[k]
                        chemin = ca[::-1] + cb[1:]
                        if not np.allclose(chemin[0], qa):
                            chemin = chemin[::-1]
                        return chemin
            A, pa, B, pb = B, pb, A, pa
        return None

    def raccourcir(self, chemin, largeur, phase, n_essais=120):
        ch = [np.array(q) for q in chemin]
        for _ in range(n_essais):
            if len(ch) <= 2:
                break
            i, j = sorted(self.rng.choice(len(ch), 2, replace=False))
            if j - i < 2:
                continue
            if self.segment_libre(ch[i], ch[j], largeur, phase):
                ch = ch[:i + 1] + ch[j:]
        # passe finale gloutonne : depuis chaque point, sauter le plus loin possible
        out, i = [ch[0]], 0
        while i < len(ch) - 1:
            j = len(ch) - 1
            while j > i + 1 and not self.segment_libre(ch[i], ch[j], largeur, phase):
                j -= 1
            out.append(ch[j])
            i = j
        return out

    def points_contournement(self, z_bas):
        """Chaînes de points de passage cartésiens candidates autour de la planche (listes de 1 ou
        2 points) : coins et bouts de son empreinte, élargie de la marge + la main (~20 cm de long,
        + D435), à la hauteur de transport ; deux coins d'un même côté (longer la planche) ;
        par-dessus. Le RRT seul trouve « un » chemin, souvent par-dessus un poteau de 54 cm ;
        ces chaînes donnent aussi les contournements par les côtés, et on garde le plus court."""
        from planche import EMPREINTE, HAUTEUR
        det = self.det
        po, (x, y), lac = det["posture"], det["planche_centre"][:2], det["planche_lacet"]
        L, l = EMPREINTE[po]
        c = np.array([x, y])
        u = np.array([np.cos(lac), np.sin(lac)])
        v = np.array([-u[1], u[0]])
        chaines = []
        for e in (0.11, 0.16):
            e += MARGE_PLANCHE
            coin = {(su, sv): np.r_[c + su * (L / 2 + e) * u + sv * (l / 2 + e) * v, z_bas]
                    for su in (-1, 1) for sv in (-1, 1)}
            chaines += [[p] for p in coin.values()]
            chaines += [[np.r_[c + su * (L / 2 + e) * u, z_bas]] for su in (-1, 1)]       # bouts
            chaines += [[np.r_[c + sv * (l / 2 + e) * v, z_bas]] for sv in (-1, 1)]       # flancs
            for sv in (-1, 1):                                                          # longer un côté
                chaines += [[coin[(-1, sv)], coin[(1, sv)]], [coin[(1, sv)], coin[(-1, sv)]]]
            for su in (-1, 1):                                                          # longer un bout
                chaines += [[coin[(su, -1)], coin[(su, 1)]], [coin[(su, 1)], coin[(su, -1)]]]
        haut = det["planche_centre"][2] + HAUTEUR[po] / 2
        for s_ in (-0.5, 0.0, 0.5):                     # par-dessus
            chaines.append([np.r_[c + s_ * L / 2 * u, max(z_bas, haut + MARGE_PLANCHE + 0.06)]])
        return chaines

    def chemin_libre(self, qa, qb, largeur, phase="libre", R=None, z_bas=None):
        """Chemin articulaire le plus court trouvé entre qa et qb (liste de points) ou None.
        Candidats : ligne droite ; passage par un point de contournement (si R et z_bas donnés) ;
        RRT-Connect. Chaque chemin est raccourci, on garde le moins coûteux."""
        if self.segment_libre(qa, qb, largeur, phase):
            return [qa, qb]
        candidats = []
        if R is not None:
            for chaine in self.points_contournement(z_bas):
                pts, q_prec, ok = [qa], qa, True
                for k, p in enumerate(chaine):
                    trouve = None
                    for g in (q_prec, (q_prec + qb) / 2, qb):
                        qv, ik_ok = self.ik(p, R, g, q_ref=g)
                        if ik_ok and self.libre(qv, largeur, phase) and self.segment_libre(q_prec, qv, largeur, phase):
                            trouve = qv
                            break
                    if trouve is None:
                        ok = False
                        break
                    pts.append(trouve)
                    q_prec = trouve
                if ok and self.segment_libre(q_prec, qb, largeur, phase):
                    candidats.append(pts + [qb])
        n_rrt = ESSAIS_RRT if not candidats else 1
        for _ in range(n_rrt):
            ch = self.rrt_connect(qa, qb, largeur, phase)
            if ch is not None:
                candidats.append(self.raccourcir(ch, largeur, phase))
        if not candidats:
            return None
        candidats.sort(key=cout)
        return self.raccourcir(candidats[0], largeur, phase, n_essais=40)

    # ---------------------------------------------------------------- plan complet
    def planifier(self, det, q0, retour=False):
        """Plan du pick-and-place pour la scène détectée det, depuis q0.
        Renvoie (plan, message). plan = liste d'étapes :
            {"type": "chemin", "q": [points], "largeur": l}  |  {"type": "saisir"}  |  {"type": "ouvrir"}
        """
        self.poser_scene(det)
        self.porte = None
        t0 = time.time()
        c = det["cube"]
        zc = red_z(*c[:2]) + CUBE_HALF
        t = det["cible"]
        zd = red_z(*t[:2]) + TARGET_H + CUBE_HALF + Z_LACHER
        p_au_dessus = np.r_[c[:2], zc + Z_PRISE + H_APPROCHE]
        p_prise = np.r_[c[:2], zc + Z_PRISE]
        p_dessus_cible = np.r_[t[:2], zd + H_APPROCHE]
        p_depose = np.r_[t[:2], zd]
        phi0 = np.degrees(np.arctan2(self.R_bas[1, 1], self.R_bas[0, 1]))
        a0 = (np.degrees(det["cube_lacet"]) - phi0 + 45) % 90 - 45
        prefixe = []
        if not self.libre(q0, LARGEUR_OUVERTE):
            # départ près de la table ou du cube (DAgger : état laissé par l'élève) : on remonte
            # d'abord à la verticale, marges réduites, puis on planifie normalement
            q_haut = self.degager(q0, LARGEUR_OUVERTE)
            if q_haut is None:
                return None, "pose de départ en collision"
            prefixe = [{"type": "chemin", "nom": "degagement", "q": [q0, q_haut], "largeur": LARGEUR_OUVERTE}]
            q0 = q_haut

        # porte = pose du cube dans le repère du tcp quand il est tenu (ne dépend que de R)
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, self.cube_pose[3:])
        Rc = Rc.reshape(3, 3)
        cube_sur_cible = np.r_[t[:2], zd - Z_LACHER, self.cube_pose[3:]]
        cube_depart = self.cube_pose.copy()

        def cote_prise(R, porte):
            """Au-dessus du cube -> prise -> montée, sans collision (plusieurs postures du coude)."""
            meilleur_c, raison_c = None, "IK impossible au-dessus du cube"
            for g, ref in self.graines(q0):
                q1, ok = self.ik(p_au_dessus, R, g, q_ref=ref)
                if not ok:
                    continue
                desc = self.ligne_cartesienne(q1, p_au_dessus, p_prise, R)
                if desc is None:
                    continue
                self.porte = None
                if not all(self.libre(q, LARGEUR_OUVERTE, "prise") for q in desc):
                    raison_c = "descente vers le cube en collision"
                    continue
                if not self.libre(desc[-1], LARGEUR_CUBE, "prise"):
                    raison_c = "prise en collision"
                    continue
                self.porte = porte
                ok_m = all(self.libre(q, LARGEUR_CUBE, "prise") for q in desc[::-1][1:])
                self.porte = None
                if not ok_m:
                    raison_c = "montée avec le cube en collision"
                    continue
                c_ = cout([q0, q1])
                if meilleur_c is None or c_ < meilleur_c[0]:
                    meilleur_c = (c_, desc)
                if ref is None:
                    break                                # la posture naturelle marche : on la garde
            return (meilleur_c[1] if meilleur_c else None), raison_c

        def cote_depose(R, porte, q_haut):
            meilleur_c, raison_c = None, "IK impossible au-dessus de la cible"
            for g, ref in self.graines(q_haut):
                q4, ok = self.ik(p_dessus_cible, R, g, q_ref=ref)
                if not ok:
                    continue
                dep = self.ligne_cartesienne(q4, p_dessus_cible, p_depose, R)
                if dep is None:
                    continue
                self.porte = porte
                ok_d = all(self.libre(q, LARGEUR_CUBE, "prise") for q in dep[:-1])
                self.porte = None
                if not ok_d:
                    raison_c = "dépose en collision"
                    continue
                self.cube_pose = cube_sur_cible
                ok_r = all(self.libre(q, LARGEUR_OUVERTE, "prise") for q in dep[::-1][1:])
                self.cube_pose = cube_depart
                if not ok_r:
                    raison_c = "remontée en collision"
                    continue
                c_ = cout([q_haut, q4])
                if meilleur_c is None or c_ < meilleur_c[0]:
                    meilleur_c = (c_, dep)
                if ref is None:
                    break
            return (meilleur_c[1] if meilleur_c else None), raison_c

        # 1. points clés sans collision pour chaque orientation de pince (lacet du cube à 90° près)
        candidats, raison = [], "aucune orientation de pince possible"
        for a in [a0 + 90 * k for k in (-2, -1, 0, 1, 2)]:
            R = self.R_pince(np.radians(a))
            porte = (R.T @ (self.cube_pose[:3] - p_prise), R.T @ Rc)
            descente, r1 = cote_prise(R, porte)
            if descente is None:
                raison = r1
                continue
            depose, r2 = cote_depose(R, porte, descente[0])
            if depose is None:
                raison = r2
                continue
            borne = cout([q0, descente[0]]) + 2 * cout(descente) + cout([descente[0], depose[0]]) + 2 * cout(depose)
            candidats.append((borne, a, R, porte, descente, depose))
        if not candidats:
            return None, raison
        candidats.sort(key=lambda x: x[0])

        # 2. segments libres (approche, transport) pour les candidats, du plus prometteur au moins bon
        meilleur = None
        for borne, a, R, porte, descente, depose in candidats:
            if meilleur is not None and borne >= meilleur[0]:
                break                                   # impossible de faire mieux
            q1, q4 = descente[0], depose[0]
            self.porte = porte
            ch_transport = self.chemin_libre(q1, q4, LARGEUR_CUBE, "libre", R=R,
                                            z_bas=max(p_au_dessus[2], p_dessus_cible[2]))
            self.porte = None
            if ch_transport is None:
                raison = "pas de chemin pour le transport"
                continue
            ch_approche = self.chemin_libre(q0, q1, LARGEUR_OUVERTE, R=R, z_bas=p_au_dessus[2] + 0.05)
            if ch_approche is None:
                raison = "pas de chemin vers le cube"
                continue
            remontee = depose[::-1]
            plan = prefixe + [{"type": "ouvrir"},
                    {"type": "chemin", "nom": "approche", "q": ch_approche, "largeur": LARGEUR_OUVERTE},
                    {"type": "chemin", "nom": "descente", "q": descente, "largeur": LARGEUR_OUVERTE},
                    {"type": "saisir"},
                    {"type": "chemin", "nom": "montee", "q": descente[::-1], "largeur": LARGEUR_CUBE},
                    {"type": "chemin", "nom": "transport", "q": ch_transport, "largeur": LARGEUR_CUBE},
                    {"type": "chemin", "nom": "depose", "q": depose, "largeur": LARGEUR_CUBE},
                    {"type": "ouvrir"},
                    {"type": "chemin", "nom": "remontee", "q": remontee, "largeur": LARGEUR_OUVERTE}]
            total = sum(cout(e["q"]) for e in plan if e["type"] == "chemin")
            if meilleur is None or total < meilleur[0]:
                meilleur = (total, a, plan, len(ch_transport) - 2, len(ch_approche) - 2)
        if meilleur is None:
            return None, raison
        total, a, plan, n_det_t, n_det_a = meilleur
        if retour:
            self.cube_pose = np.r_[t[:2], zd - Z_LACHER, self.cube_pose[3:]]
            ch_retour = self.chemin_libre(plan[-1]["q"][-1], q0, LARGEUR_OUVERTE)
            if ch_retour is None:
                return None, "pas de chemin pour revenir à la pose de départ"
            plan.append({"type": "chemin", "nom": "retour", "q": ch_retour, "largeur": LARGEUR_OUVERTE})
        self.porte = None
        msg = (f"plan OK en {time.time() - t0:.1f} s : pince {a:+.0f}°, chemin {total:.2f} rad, "
               f"détours : approche {n_det_a} pt, transport {n_det_t} pt")
        return plan, msg


    def degager(self, q, largeur):
        """Remonte la pince à la verticale (jusqu'à 8 cm au-dessus de la table, marges réduites) pour
        sortir d'une pose trop proche de la table / du cube. Renvoie q en haut, ou None."""
        m, d, ix = self.model, self.data, self.ix
        self._poser_bras(q, largeur)
        mujoco.mj_kinematics(m, d)
        p = d.site_xpos[ix.tcp].copy()
        R = d.site_xmat[ix.tcp].reshape(3, 3).copy()
        z_haut = red_z(*p[:2]) + CUBE_HALF + Z_PRISE + H_APPROCHE + 0.02
        if p[2] >= z_haut:
            return None
        ligne = self.ligne_cartesienne(q, p, np.r_[p[:2], z_haut], R, n=3)
        if ligne is None or not all(self.libre(x, largeur, "prise") for x in ligne[1:]):
            return None
        return ligne[-1] if self.libre(ligne[-1], largeur) else None

    def planifier_transport(self, det, q, cube_pose):
        """Plan quand le cube est DÉJÀ tenu (DAgger) : (montée) -> au-dessus de la cible -> dépose ->
        lâcher -> remontée, en gardant l'orientation actuelle de la pince."""
        m, d, ix = self.model, self.data, self.ix
        self.poser_scene(det)
        self._poser_bras(q, LARGEUR_CUBE)
        mujoco.mj_kinematics(m, d)
        p, Rt = d.site_xpos[ix.tcp].copy(), d.site_xmat[ix.tcp].reshape(3, 3).copy()
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, cube_pose[3:])
        porte = (Rt.T @ (cube_pose[:3] - p), Rt.T @ Rc.reshape(3, 3))
        phi = np.arctan2(Rt[1, 1], Rt[0, 1]) - np.arctan2(self.R_bas[1, 1], self.R_bas[0, 1])
        R = self.R_pince(phi)                            # pince remise bien verticale, même lacet
        t = det["cible"]
        zd = red_z(*t[:2]) + TARGET_H + CUBE_HALF + Z_LACHER + (p[2] - cube_pose[2])   # tcp au lâcher
        p_dessus, p_depose = np.r_[t[:2], zd + H_APPROCHE], np.r_[t[:2], zd]
        self.porte = porte
        prefixe, q_dep = [], np.asarray(q, float)
        if not self.libre(q_dep, LARGEUR_CUBE):
            z_haut = red_z(*p[:2]) + CUBE_HALF + Z_PRISE + H_APPROCHE + 0.02
            ligne = self.ligne_cartesienne(q_dep, p, np.r_[p[:2], z_haut], R, n=3)
            if (ligne is None or not all(self.libre(x, LARGEUR_CUBE, "prise") for x in ligne[1:])
                    or not self.libre(ligne[-1], LARGEUR_CUBE)):
                self.porte = None
                return None, "cube tenu : impossible de remonter"
            prefixe = [{"type": "chemin", "nom": "montee", "q": ligne, "largeur": LARGEUR_CUBE}]
            q_dep = ligne[-1]
        meilleur = None
        for g, ref in self.graines(q_dep, n=6):
            q4, ok = self.ik(p_dessus, R, g, q_ref=ref)
            if not ok or not self.libre(q4, LARGEUR_CUBE):
                continue
            dep = self.ligne_cartesienne(q4, p_dessus, p_depose, R)
            if dep is None or not all(self.libre(x, LARGEUR_CUBE, "prise") for x in dep[:-1]):
                continue
            meilleur = dep
            break
        if meilleur is None:
            self.porte = None
            return None, "cube tenu : pas de dépose possible"
        ch = self.chemin_libre(q_dep, meilleur[0], LARGEUR_CUBE, "libre", R=R, z_bas=p_dessus[2])
        self.porte = None
        if ch is None:
            return None, "cube tenu : pas de chemin vers la cible"
        plan = prefixe + [{"type": "chemin", "nom": "transport", "q": ch, "largeur": LARGEUR_CUBE},
                          {"type": "chemin", "nom": "depose", "q": meilleur, "largeur": LARGEUR_CUBE},
                          {"type": "ouvrir"},
                          {"type": "chemin", "nom": "remontee", "q": meilleur[::-1], "largeur": LARGEUR_OUVERTE}]
        return plan, "plan depuis la prise"


def _profil(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5       # quintique : vitesse et accélération nulles aux bouts


def densifier(plan, dq=DQ_EXPERT, ouverte=True):
    """Plan -> suite de consignes (q, pince ouverte ?, précis ?), une par pas d'env (40 ms).
    « précis » = segment vertical près du cube / de la table ou attente de la pince (pas de bruit DART).
    Chaque chemin est parcouru avec un profil quintique le long de la ligne brisée : départ et
    arrivée à vitesse nulle, sans à-coup (sinon le cube glisse dans la pince), vitesse maxi dq/pas."""
    refs, q_cour, ouverte = [], None, True
    for e in plan:
        if e["type"] == "chemin":
            pts = [np.asarray(q, float) for q in e["q"]]
            if q_cour is None:
                q_cour = pts[0]
            pts[0] = q_cour
            longueurs = [np.abs(b_ - a_).max() for a_, b_ in zip(pts[:-1], pts[1:])]
            S = float(sum(longueurs))
            if S < 1e-6:
                continue
            cum = np.r_[0, np.cumsum(longueurs)]
            precis = e["nom"] in ("descente", "montee", "depose", "remontee")
            n = max(2, int(np.ceil(1.875 * S / dq)))       # 1,875 = vitesse max / vitesse moyenne
            for k in range(1, n + 1):
                sk = _profil(k / n) * S
                i = min(int(np.searchsorted(cum, sk, side="right")) - 1, len(longueurs) - 1)
                f = (sk - cum[i]) / longueurs[i] if longueurs[i] > 0 else 1.0
                refs.append((pts[i] + (pts[i + 1] - pts[i]) * min(f, 1.0), ouverte, precis))
            q_cour = pts[-1]
        elif e["type"] == "saisir":
            ouverte = False
            refs += [(q_cour, False, True)] * PAS_FERMETURE
        elif e["type"] == "ouvrir":
            ouverte = True
            if q_cour is not None:
                refs += [(q_cour, True, True)] * PAS_OUVERTURE
    return refs


def vers_etapes(plan):
    """Plan (planifié avec retour=True) -> étapes du jumeau, même format que jumeau/planif_expert.py
    (vers_json, puis E dans MuJoCo et R sur le vrai robot). Chaque point du chemin devient un point de
    passage articulaire du MotionGenerator de libfranka (il s'arrête sur chacun : chemins courts)."""
    etapes = []
    for e in plan:
        if e["type"] == "chemin":
            pts = e["q"][1:]
            if e["nom"] in ("descente", "montee", "depose", "remontee"):
                pts = pts[-1:]          # 6 cm à la verticale : un seul point (pas d'arrêts en route)
            for k, q in enumerate(pts):
                etapes.append({"type": "q", "nom": f"{e['nom']} {k + 1}", "q": np.asarray(q)})
        elif e["type"] == "saisir":
            etapes.append({"type": "saisir", "largeur": LARGEUR_CUBE, "force": 20.0})
        else:
            etapes.append({"type": "ouvrir", "largeur": LARGEUR_OUVERTE})
    return etapes


class ExpertPlanche:
    """Expert « suiveur » : planifie au début de l'épisode, puis à chaque pas donne l'action qui amène
    la consigne articulaire de l'env sur le point suivant du plan. S'il est écarté (bruit DART), il
    revient sur le plan (l'action dépend de l'état, pas seulement du temps)."""

    def __init__(self, seed=0):
        self.pl = Planificateur(seed)
        self.refs = None
        self.plan, self.message = None, ""
        self.precis = True          # True quand l'expert est près du cube / de la table (pas de bruit)

    def reset(self, env, retour=False):
        self.pas_sans_plan = -1
        det = env.detection()
        # tirages du RRT déterminés par la scène : même scène -> même plan (l'imitation apprend une
        # fonction scène -> mouvement ; un expert qui change d'avis d'une fois sur l'autre la brouille)
        cle = np.round(np.r_[det["cube"], det["cube_lacet"], det["cible"], det["planche_centre"],
                             det["planche_lacet"], env.q_cmd], 4)
        self.pl.rng = np.random.default_rng(abs(hash(tuple(cle.tolist()))) % 2 ** 32)
        self.plan, self.message = self.pl.planifier(det, env.q_cmd.copy(), retour=retour)
        self.refs = densifier(self.plan) if self.plan else None
        self.k = 0
        if self.refs is not None:
            self._reperes(env.data.xpos[env.ix.cube_b])
        return self.plan is not None

    # ------------------------------------------------------------ DAgger : étiqueter n'importe quel état
    def oublier(self):
        """Début d'un épisode DAgger : pas de plan en cours (il sera fait au premier appel)."""
        self.plan, self.refs, self.pas_sans_plan = None, None, -1

    def etiquette(self, env):
        """Action de l'expert pour l'état ACTUEL de env, quel que soit le chemin qui y a mené (états
        visités par l'élève, DAgger). Suit le plan en cours tant que l'état en reste proche, sinon
        replanifie depuis l'état actuel (cube tenu ou non). None si aucune bonne action n'existe
        (cube renversé, aucun plan) : l'état n'est alors pas étiqueté."""
        d, ix = env.data, env.ix
        cube = d.xpos[ix.cube_b].copy()
        if env.etat_reussi():
            self.refs = None
            return np.r_[np.zeros(7), 1.0].astype(np.float32)
        tcp = d.site_xpos[ix.tcp]
        largeur = d.qpos[ix.f1] + d.qpos[ix.f2]
        ferme = d.ctrl[7] < 100
        tenu = ferme and np.linalg.norm(tcp - cube) < 0.02 and 0.03 < largeur < 0.045
        if not tenu and abs(d.xmat[ix.cube_b].reshape(3, 3)[2, 2]) < np.cos(np.radians(10)):
            return None                                          # cube renversé sur la table
        self.replanifie = False
        if self.refs is not None and self._plan_encore_bon(env, cube, tenu):
            return self.action(env)
        self.replanifie = True
        if env.steps < getattr(self, "pas_sans_plan", -1):
            return None                    # dernier essai raté il y a moins de 10 pas : on n'insiste pas
        det = env.detection()
        if tenu:
            cube_pose = np.r_[cube, d.xquat[ix.cube_b]]
            self.plan, self.message = self.pl.planifier_transport(det, env.q_cmd.copy(), cube_pose)
        else:
            self.plan, self.message = self.pl.planifier(det, env.q_cmd.copy())
        if self.plan is None:
            self.refs = None
            self.pas_sans_plan = env.steps + 10
            return None
        self.refs, self.k = densifier(self.plan, ouverte=not tenu), 0
        self._reperes(cube)
        return self.action(env)

    def _reperes(self, cube):
        """Index du premier pas « pince fermée » du plan (avant : le cube doit être à sa place)."""
        self.cube_plan = cube.copy()
        self.k_saisie = next((i for i, r in enumerate(self.refs) if not r[1]), -1)

    def _plan_encore_bon(self, env, cube, tenu):
        if self.k >= len(self.refs):
            return False
        q_ref, ouverte, _ = self.refs[self.k]
        avant_prise = self.k < self.k_saisie
        if avant_prise and not tenu and np.linalg.norm(cube - self.cube_plan) > 0.01:
            return False                                         # cube bousculé avant la prise
        if tenu and ouverte:
            return False                                         # le plan ne s'attend pas à tenir le cube
        if not tenu and not ouverte and env.data.ctrl[7] < 100:
            # le plan croit tenir le cube : normal pendant la fermeture (attente), pas après
            fermeture = self.k > 0 and self.refs[self.k - 1][1] is False and np.allclose(self.refs[self.k - 1][0], q_ref)
            if not fermeture:
                return False
        if np.abs(q_ref - env.q_cmd).max() <= DQ_MAX * 1.2:
            return True                                          # prochain point atteignable : on suit
        # l'élève s'est écarté : on se recale sur le point du plan le plus proche (un peu en arrière
        # ou en avant) et on vise le suivant ; trop loin du plan -> replanifier
        j0, j1 = max(0, self.k - 5), min(len(self.refs), self.k + 25)
        ecarts = [np.abs(self.refs[j][0] - env.q_cmd).max() for j in range(j0, j1)]
        j = j0 + int(np.argmin(ecarts))
        if min(ecarts) > 0.03 or self.refs[j][1] != ouverte:
            return False
        self.k = min(j + 1, len(self.refs) - 1)
        return True

    def fini(self):
        return self.refs is None or self.k >= len(self.refs)

    def action(self, env):
        if self.refs is None:
            self.precis = True
            return np.r_[np.zeros(7), 1.0].astype(np.float32)
        q_ref, ouverte, self.precis = self.refs[min(self.k, len(self.refs) - 1)]
        a = np.clip((q_ref - env.q_cmd) / DQ_MAX, -1, 1)
        if self.k < len(self.refs):
            # le point sera atteint après cette action (si elle n'est pas trop perturbée)
            if np.abs(q_ref - env.q_cmd).max() <= DQ_MAX * 1.2:
                self.k += 1
        return np.r_[a, 1.0 if ouverte else -1.0].astype(np.float32)


# ------------------------------------------------------------------------------- tests
def jouer_episode(env, expert, seed, view=False, options=None):
    env.reset(seed=seed, options=options)
    if not expert.reset(env):
        return {"plan": False, "message": expert.message, "succes": False, "collision": "", "pas": 0}
    info = {}
    for _ in range(400):
        _, _, term, trunc, info = env.step(expert.action(env))
        if view:
            time.sleep(0.02)
        if term or trunc or (expert.fini() and info["is_success"]):
            break
    return {"plan": True, "message": expert.message, "succes": info["is_success"],
            "collision": info["collision"], "pas": info["pas"], "longueur_q": info["longueur_q"]}


def main():
    view = "--view" in sys.argv
    n = int(sys.argv[sys.argv.index("--view" if view else "--test") + 1]) if len(sys.argv) > 2 else 10
    env = TwinPlancheEnv(render_mode="human" if view else None)
    expert = ExpertPlanche()
    stats = {}
    for s in range(n):
        t0 = time.time()
        r = jouer_episode(env, expert, s, view)
        sc = env.scene
        cle = sc["posture"]
        st = stats.setdefault(cle, [0, 0, 0])
        st[0] += 1
        st[1] += r["plan"]
        st[2] += r["succes"]
        etat = "REUSSI" if r["succes"] else ("pas de plan" if not r["plan"] else
                                              f"ECHEC {r['collision']}")
        print(f"{s:3d} {sc['posture']:8s} {'chemin' if sc['sur_chemin'] else 'ailleurs':8s} "
              f"{etat:28s} {r['pas']:3d} pas  {time.time() - t0:4.1f} s  | {r['message']}")
    print()
    for k, (a, b, c) in stats.items():
        print(f"{k:8s} : {a} scènes, plan trouvé {b}, réussies {c}")
    env.close()


if __name__ == "__main__":
    main()
