"""Environnement Gymnasium du JUMEAU avec obstacle : pick-and-place sur la vraie scène du labo
(tables, trou, cube 23,3 g tourné au hasard, cible) + la planche d'aggloméré (planche.py),
couchée, sur la tranche ou debout, placée au hasard.

Différences avec common/panda_pick_env.py (scène générique, qui reste intacte) :
  - scène twin calibrée (twin_scene.py), bras avec compensation de gravité (comme le vrai Panda) ;
  - ACTIONS ARTICULAIRES : 7 petits déplacements d'angle (au plus DQ_MAX rad par pas) + pince.
    Avec un poteau de 54 cm, c'est tout le bras (coude, avant-bras, D435) qui doit l'éviter, pas
    seulement la pince ; et la trajectoire produite est directement une suite d'angles, rejouable
    par le vrai robot (même chemin que la boucle E/R du jumeau) ;
  - observation = ce que donnerait la PERCEPTION (position/lacet du cube, cible, planche) avec un
    biais de détection tiré à chaque épisode (~2 mm / 3°, mesuré le 23/11) + la proprioception ;
  - toucher la planche (bras, D435 ou cube tenu) ou la table avec le bras = échec immédiat ;
  - coût de trajet : chaque radian parcouru coûte un peu -> la meilleure politique va tout droit
    quand le chemin est libre et ne fait un détour que si la planche gêne.

Observation (40 valeurs, voir _obs) ; action (8 valeurs dans [-1, 1]) : dq1..dq7, pince (>0 = ouverte).

    python avec_obstacle/common/twin_env.py --view            -> 5 scènes aléatoires dans le viewer (bras immobile)
    python avec_obstacle/common/twin_env.py --test 200        -> statistiques du tirage des scènes
"""
import os
import sys

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))   # twin_scene (scène du labo)
from planche import (HAUTEUR, POSTURES, ajouter_planche, pose_planche, tirer_scene)  # noqa: E402
from twin_scene import (CUBE_HALF, HOME_Q, TARGET_H, WEBCAM, build_twin_spec, red_z)  # noqa: E402

TCP_OFFSET = 0.1034           # bride -> centre des doigts (EE « Franka Hand » de Desk)
FRAME_SKIP = 20               # 20 x 2 ms = 40 ms par pas (25 Hz, comme l'ancien env)
DQ_MAX = 0.05                 # rad par pas au plus (1,25 rad/s) ; le vrai robot rejoue plus lentement
MAX_STEPS = 400               # 16 s par épisode (contournements longs, départs et arrêts en douceur)
SUCCES_DIST = 0.02            # cube à moins de 2 cm de la cible (horizontal), posé et lâché
LARGEUR_LACHE = 0.045
BRUIT_START = 0.15            # rad : la pose de départ varie autour de HOME_Q
# biais de détection tirés à chaque épisode (écart-type), d'après les mesures du 23/11
BRUIT = {"cube_xy": 0.0015, "cube_lacet": np.radians(2.5), "cible_xy": 0.002,
         "planche_xy": 0.003, "planche_lacet": np.radians(1.5)}

CORPS_ROBOT = ["link1", "link2", "link3", "link4", "link5", "link6", "link7",
               "hand", "left_finger", "right_finger"]
DOIGTS = ["left_finger", "right_finger"]
OBS_DIM = 40
OBS_PLANCHE = slice(32, 40)      # les 8 valeurs qui décrivent la planche (masquées pour l'ablation)


def construire_spec():
    """Scène twin + planche + site tcp + gravcomp + boîtier D435 solide (il peut taper la planche)."""
    spec = build_twin_spec()
    for n in CORPS_ROBOT:
        spec.body(n).gravcomp = 1.0
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.004, 0, 0], rgba=[1, 0, 0, 0])
    g = spec.geom("d435_boitier")
    g.contype, g.conaffinity = 1, 1
    # pince : le modèle menagerie ne serre qu'environ 2 N : le cube glisse dans les virages (jusqu'à
    # le lâcher). Le vrai grasp du jumeau serre à 20 N. Force du tendon = g.ctrl + b0 - 100.L :
    # fermée (ctrl 0) ~ -12 N au contact du cube, ouverte (ctrl 255) : équilibre à L = 4 cm comme avant.
    # Contacts du cube raidis (sinon les doigts s'enfoncent de plusieurs mm dans un cube de 23 g).
    act = spec.actuator("actuator8")
    gp, bp = list(act.gainprm), list(act.biasprm)
    gp[0], bp[0] = 14.0 / 255.0, -10.0
    act.gainprm, act.biasprm = gp, bp
    cg = spec.geom("cube_geom")
    cg.solref = [0.004, 1.0]
    cg.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    ajouter_planche(spec)
    return spec


def lacet_depuis_quat(q):
    """Lacet (rad) de l'axe x d'un corps dont le quaternion est q (w, x, y, z)."""
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Indices:
    """Adresses utiles dans un modèle construit par construire_spec() (partagé avec l'expert)."""

    def __init__(self, m):
        self.qadr = np.array([m.jnt_qposadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.dadr = np.array([m.jnt_dofadr[m.joint(f"joint{i}").id] for i in range(1, 8)])
        self.f1 = m.jnt_qposadr[m.joint("finger_joint1").id]
        self.f2 = m.jnt_qposadr[m.joint("finger_joint2").id]
        self.cube_q = m.jnt_qposadr[m.joint("cube_joint").id]
        self.cube_d = m.jnt_dofadr[m.joint("cube_joint").id]
        self.pl_mocap = m.body_mocapid[m.body("planche").id]
        self.cube_b = m.body("cube").id
        self.pl_b = m.body("planche").id
        self.cible_mocap = m.body_mocapid[m.body("target").id]
        self.tcp = m.site("tcp").id
        ids = [m.joint(f"joint{i}").id for i in range(1, 8)]
        self.q_lo, self.q_hi = m.jnt_range[ids, 0].copy(), m.jnt_range[ids, 1].copy()
        self.robot_b = {m.body(n).id for n in CORPS_ROBOT}
        self.doigts_b = {m.body(n).id for n in DOIGTS}
        self.cube_g = m.geom("cube_geom").id
        self.pl_g = m.geom("planche_geom").id
        self.cible_g = m.geom("target_mark").id
        tr = m.body("table_rouge").id
        self.table_g = {g for g in range(m.ngeom) if m.geom_bodyid[g] == tr} | {m.geom("table_blanche").id}


class TwinPlancheEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 25}

    def __init__(self, render_mode=None, obs_planche=True, bruit=True, postures=None):
        """obs_planche=False : la planche est cachée à la politique (test d'ablation).
        postures : sous-ensemble de ("plat", "tranche", "debout") à tirer (toutes par défaut)."""
        self.model = construire_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.ix = Indices(self.model)
        self.render_mode = render_mode
        self.obs_planche = obs_planche
        self.bruit = bruit
        self.postures = tuple(postures or POSTURES)
        self.viewer = None
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float64)
        self.jacp = np.zeros((3, self.model.nv))
        m = self.model
        self.avance = -m.actuator_biasprm[:7, 2] / m.actuator_gainprm[:7, 0]     # kv / kp (0,1 s)
        self.scene = None
        self.cam_pos = np.array(WEBCAM["pos"], float)

    # ------------------------------------------------------------------ scène
    def poser_scene(self, sc, q0):
        """Place bras (q0, pince ouverte), cube, cible, planche (fixe) ; le cube se pose 0,3 s."""
        m, d, ix = self.model, self.data, self.ix
        mujoco.mj_resetData(m, d)
        d.qpos[ix.qadr] = q0
        d.qpos[ix.f1] = d.qpos[ix.f2] = 0.04
        d.ctrl[:7] = q0
        d.ctrl[7] = 255.0
        cx, cy = sc["cube_xy"]
        a = np.radians(sc["cube_lacet_deg"])
        d.qpos[ix.cube_q:ix.cube_q + 7] = [cx, cy, red_z(cx, cy) + CUBE_HALF + 0.0005,
                                           np.cos(a / 2), 0, 0, np.sin(a / 2)]
        tx, ty = sc["cible_xy"]
        d.mocap_pos[ix.cible_mocap] = [tx, ty, red_z(tx, ty)]
        pq = pose_planche(sc["posture"], *sc["planche_xy"], sc["planche_lacet"])
        d.mocap_pos[ix.pl_mocap], d.mocap_quat[ix.pl_mocap] = pq[:3], pq[3:]
        mujoco.mj_forward(m, d)
        mujoco.mj_step(m, d, nstep=150)                    # le cube se pose
        self.planche_ref = (pq[:3].copy(), pq[3:].copy())
        self.cube_z0 = d.xpos[ix.cube_b][2]

    def _visible_webcam(self, point):
        """True si la planche ne cache pas ce point à la webcam (rayon webcam -> point)."""
        vec = np.asarray(point, float) - self.cam_pos
        geomid = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(self.model, self.data, self.cam_pos, vec, None, 1, -1, geomid)
        return not (geomid[0] == self.ix.pl_g and 0 <= dist < 0.999)

    def contacts_interdits(self):
        """Liste des contacts interdits : robot (ou cube tenu) / planche, robot / tables."""
        m, d, ix = self.model, self.data, self.ix
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            for ga, ba, gb in ((g1, b1, g2), (g2, b2, g1)):
                if gb == ix.pl_g and (ba in ix.robot_b or ga == ix.cube_g):
                    out.append(f"{m.body(ba).name} / planche")
                elif ba in ix.robot_b and gb in ix.table_g and c.dist < -0.0005:
                    out.append(f"{m.body(ba).name} / {m.geom(gb).name}")
        return out

    def scene_ok(self, sc, q0):
        """Scène utilisable : cube et cible vus par la webcam (pas cachés par la planche), bras de
        départ sans contact."""
        self.poser_scene(sc, q0)
        if self.contacts_interdits():
            return False
        d, ix = self.data, self.ix
        top = d.xpos[ix.cube_b] + [0, 0, CUBE_HALF]
        cible = d.mocap_pos[ix.cible_mocap] + [0, 0, TARGET_H]
        return self._visible_webcam(top) and self._visible_webcam(cible)

    # ------------------------------------------------------------------ Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        options = options or {}
        for _ in range(200):
            if "scene" in options:
                sc = dict(options["scene"])
            else:
                sc = tirer_scene(rng, posture=self.postures[rng.integers(len(self.postures))])
            q0 = np.array(options.get("q0", HOME_Q + rng.uniform(-BRUIT_START, BRUIT_START, 7)))
            q0 = np.clip(q0, self.ix.q_lo + 0.05, self.ix.q_hi - 0.05)
            if self.scene_ok(sc, q0) or "scene" in options:
                break
        self.scene, self.q0 = sc, q0
        # biais de détection de CET épisode (la vraie webcam se trompe toujours un peu)
        b = {k: (rng.normal(0, s, 2) if k.endswith("xy") else rng.normal(0, s)) if self.bruit else
             (np.zeros(2) if k.endswith("xy") else 0.0) for k, s in BRUIT.items()}
        self.biais = b
        self.q_cmd = q0.copy()
        self.steps = 0
        self.longueur_q = 0.0
        self.collision = ""
        self.succes_vu = False
        if self.render_mode == "human":
            self.render()
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        m, d, ix = self.model, self.data, self.ix
        q_avant = d.qpos[ix.qadr].copy()
        q_cmd_avant = self.q_cmd.copy()
        self.q_cmd = np.clip(self.q_cmd + a[:7] * DQ_MAX, ix.q_lo, ix.q_hi)
        # anticipation de vitesse : les actionneurs PD du modèle menagerie (couple = kp.(ctrl - q) - kv.dq)
        # traînent de kv/kp = 0,1 s derrière une consigne qui avance (0,1 rad à 1 rad/s, soit ~6 cm au
        # bout du bras). Le vrai Panda suit sa trajectoire au dixième de mm : on décale la consigne de
        # kv/kp x vitesse pour que le bras simulé soit SUR la consigne, comme le vrai.
        v_cmd = (self.q_cmd - q_cmd_avant) / (FRAME_SKIP * m.opt.timestep)
        d.ctrl[:7] = np.clip(self.q_cmd + self.avance * v_cmd, ix.q_lo, ix.q_hi)
        d.ctrl[7] = 255.0 if a[7] > 0 else 0.0
        contacts = []
        for _ in range(2):                                 # contrôle des contacts toutes les 20 ms
            mujoco.mj_step(m, d, nstep=FRAME_SKIP // 2)
            contacts += self.contacts_interdits()
        self.steps += 1
        self.longueur_q += float(np.abs(d.qpos[ix.qadr] - q_avant).sum())

        succes = self.etat_reussi()
        terminated = False
        if contacts:
            self.collision = contacts[0]
            terminated = True
            reward = -10.0
            succes = False
        else:
            reward = self._reward(succes) - 0.05 * float(np.abs(a[:7]).mean())
        truncated = self.steps >= MAX_STEPS
        if self.render_mode == "human":
            self.render()
        info = {"is_success": bool(succes), "collision": self.collision,
                "longueur_q": self.longueur_q, "pas": self.steps}
        return self._obs(), reward, terminated, truncated, info

    def etat_reussi(self):
        d, ix = self.data, self.ix
        cube = d.xpos[ix.cube_b]
        cible = d.mocap_pos[ix.cible_mocap]
        largeur = d.qpos[ix.f1] + d.qpos[ix.f2]
        z_pose = red_z(*cible[:2]) + TARGET_H + CUBE_HALF
        return bool(np.hypot(*(cube[:2] - cible[:2])) < SUCCES_DIST and abs(cube[2] - z_pose) < 0.01
                    and largeur > LARGEUR_LACHE)

    def _reward(self, succes):
        """Même récompense étagée que l'ancien env (approche < prise < transport < réussite),
        sans creux entre les étapes. La collision (-10, fin) et le coût de trajet sont dans step()."""
        if succes:
            return 3.0
        d, ix = self.data, self.ix
        tcp, cube = d.site_xpos[ix.tcp], d.xpos[ix.cube_b]
        cible = d.mocap_pos[ix.cible_mocap] + [0, 0, TARGET_H + CUBE_HALF]
        d_grip = np.linalg.norm(cube - tcp)
        d_goal = np.linalg.norm(cible - cube)
        largeur = d.qpos[ix.f1] + d.qpos[ix.f2]
        tenu = d_grip < 0.02 and 0.03 < largeur < LARGEUR_LACHE
        near = 0.5 * (1 - np.tanh(3 * d_grip)) + 0.5 * (1 - np.tanh(15 * d_grip))
        r = 0.25 * near
        if tenu:
            far = 0.5 * (1 - np.tanh(2 * d_goal)) + 0.5 * (1 - np.tanh(10 * d_goal))
            r += 0.25 + 1.0 * far
        return float(r)

    # ------------------------------------------------------------------ perception simulée
    def detection(self):
        """Ce que la webcam donnerait : positions RÉELLES (sim) + biais de l'épisode.
        Le cube suit la sim (dans le jumeau, la politique joue en sim avant le vrai robot)."""
        d, ix, b = self.data, self.ix, self.biais
        cube = d.xpos[ix.cube_b].copy()
        cube[:2] += b["cube_xy"]
        lacet = lacet_depuis_quat(d.xquat[ix.cube_b]) + b["cube_lacet"]
        cible = d.mocap_pos[ix.cible_mocap].copy()
        cible[:2] += b["cible_xy"]
        p0, q0 = self.planche_ref
        pl = p0.copy()
        pl[:2] += b["planche_xy"]
        sc = self.scene
        return {"cube": cube, "cube_lacet": lacet, "cible": cible, "planche_centre": pl,
                "planche_lacet": sc["planche_lacet"] + b["planche_lacet"], "posture": sc["posture"]}

    def _obs(self):
        d, ix = self.data, self.ix
        det = self.detection()
        q = d.qpos[ix.qadr]
        tcp = d.site_xpos[ix.tcp].copy()
        R = d.site_xmat[ix.tcp].reshape(3, 3)
        phi = np.arctan2(R[1, 1], R[0, 1])                    # direction de fermeture des doigts
        largeur = d.qpos[ix.f1] + d.qpos[ix.f2]
        rel = det["cube_lacet"] - phi                          # lacet cube / pince, à 90° près
        cible = det["cible"] + [0, 0, TARGET_H + CUBE_HALF]
        if self.obs_planche:
            po = det["posture"]
            c = det["planche_centre"]
            pl = np.r_[c[:2], c[2] + HAUTEUR[po] / 2, np.cos(2 * det["planche_lacet"]),
                       np.sin(2 * det["planche_lacet"]), [float(po == p) for p in POSTURES]]
        else:
            pl = np.zeros(8)
        return np.concatenate([
            q, tcp, [largeur],                                 # 7 + 3 + 1
            d.qvel[ix.dadr],                                   # 7 : où en est le mouvement
            det["cube"], [np.cos(4 * rel), np.sin(4 * rel)],   # 3 + 2
            det["cube"] - tcp, cible, cible - det["cube"],     # 3 + 3 + 3
            pl,                                                # 8 : centre x, y, haut, lacet (x2), posture
        ])

    # ------------------------------------------------------------------ affichage
    def render(self):
        if self.viewer is None:
            from mujoco import viewer
            try:
                self.viewer = viewer.launch_passive(self.model, self.data, show_left_ui=False,
                                                    show_right_ui=False)
            except TypeError:
                self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def _test(n):
    import time
    env = TwinPlancheEnv()
    t0 = time.time()
    comptes = {}
    for k in range(n):
        env.reset(seed=k)
        cle = (env.scene["posture"], env.scene["sur_chemin"])
        comptes[cle] = comptes.get(cle, 0) + 1
    print(f"{n} scènes en {time.time() - t0:.1f} s")
    for (po, ch), v in sorted(comptes.items()):
        print(f"  {po:8s} {'sur le chemin' if ch else 'ailleurs':14s} {v}")
    t0 = time.time()
    for _ in range(200):
        env.step(np.zeros(8))
    print(f"200 pas immobiles : {time.time() - t0:.2f} s, collision : '{env.collision}'")


if __name__ == "__main__":
    if "--test" in sys.argv:
        _test(int(sys.argv[sys.argv.index("--test") + 1]))
    elif "--view" in sys.argv:
        import time
        env = TwinPlancheEnv(render_mode="human")
        for k in range(5):
            env.reset(seed=k)
            print(f"scène {k} : {env.scene['posture']}, {'sur le chemin' if env.scene['sur_chemin'] else 'ailleurs'}")
            for _ in range(75):
                env.step(np.r_[np.zeros(7), 1.0])
                time.sleep(0.04)
        env.close()
