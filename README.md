# Digital twin of a Franka Panda: pick-and-place checked in simulation, then run on the real robot

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-0A7BBB)
![libfranka](https://img.shields.io/badge/libfranka-0.8.0%20(C%2B%2B)-555555)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-SAC-6A5ACD)
![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)

![obstacle](media/obstacle.gif)

*Left: the real bench seen by a webcam. Right: its digital twin in MuJoCo, updated live. A plank is placed on the
table, the twin picks it up from the cameras, a path around it is chosen and played in the twin, and only then does
the real Franka Panda replay the same motion. [Full video](media/obstacle.mp4) (the ~70 s spent planning are cut).*

A **digital twin** of a real Franka Panda and its lab bench, built in MuJoCo: the robot, the tables (including
their 2° slope and a cable hole), the cameras, the cube and the target, all calibrated in the robot's own frame.
The twin is kept in sync with reality (a webcam places the cube and the target, the virtual arm copies the real joint
angles), and every motion is **played in the twin first**: the real robot only executes it if the simulation succeeds
and nothing has moved since. The webcam then measures what really happened, so plan and outcome can be compared.

An obstacle can be added: a 54 cm plank, lying flat, on its edge or standing, anywhere on the table. A small
network decides which way to go around it, and the same check-then-execute loop applies.

The project also compares two ways of learning the task, **imitation learning** from a scripted expert and
**reinforcement learning** from a reward. They are trained so far in a generic simulated scene; the next step is to
train them in the calibrated twin and run them on the robot through the same check-then-execute loop.

This is my final-year engineering project (PFE) at Polytech Montpellier, with the company EuroDAO:
*Digital twin and AI for autonomous robotic manipulation with a Franka Panda*.
MuJoCo is used on a laptop CPU while the Isaac Sim workstation planned for the project is not available.

| | |
|---|---|
| Twin vs real robot, same joint angles | fingertip positions within **0.002 mm** |
| Cube detected by the webcam (vs the robot as ground truth) | **1.8 mm**, yaw **3.2°** (mean, cross-validated) |
| Real pick-and-place, cube to target | **1.0–4.1 mm** (5 measured runs) |
| Learning in simulation, 50 unseen episodes | behavior cloning from 2 cameras **50/50**, SAC **49/50** |
| With the plank (91 scenes in the twin) | route chosen by IL **92%**, by RL **85%**, full planner 97% |
| Real pick-and-place with the plank | 7 runs, cube **2.4–8.2 mm** from the target, no collision |

**Contents**: [the digital twin on the real robot](#digital-twin-on-the-real-robot) ·
[obstacle avoidance](#obstacle-avoidance-a-plank-on-the-table) ·
[learning in simulation (IL vs RL)](#learning-in-simulation-imitation-vs-reinforcement-learning) ·
[simulated RGB-D perception](#simulated-rgb-d-perception) · [repository layout](#repository-layout) ·
[quick start](#quick-start) · [next steps](#next-steps)

## Digital twin on the real robot

![digital twin](media/digital_twin.gif)

*First version, without obstacle (23/11/2026). [Full video](media/digital_twin.mp4).*

![setup](media/setup.jpg)

The bench: a Franka Panda on its own small table, a work table next to it, a blue 40 mm cube and a green 60 mm
target (both 3D-printed), an AprilTag glued on the table, a webcam on a tripod looking down at the scene, and a
RealSense D435 on the wrist.

| 1. Live update | 2. Check in the twin (key E) | 3. Execute on the robot (key R) |
|---|---|---|
| The webcam detects the cube and the target ~6 times per second; they move in the twin as they move on the table. The virtual arm copies the real joint angles. | The scripted expert plans the pick-and-place from the real arm pose and plays it in MuJoCo: inverse kinematics, collisions with the tables, grasp, placement. | Only if the simulation succeeded and nothing moved since: the real Panda replays **the same joint waypoints**, slowly. The twin follows the real arm at 25 Hz, then the webcam measures where the cube really ended up. |

### Calibrating the twin

Everything is expressed in the robot base frame and **measured with the robot itself**: a pencil held in the
gripper is brought onto points of the scene, and the robot reports where its tip is (pencil length recovered by also
touching the robot's own table, consistent to ±0.7 mm).

![pencil calibration](media/calib_pencil.jpg)

- **Webcam**: intrinsics by Zhang's method (45 checkerboard images, 0.63 px). For its pose, an AprilTag (tag36h11,
  80 mm) is glued on the table; its 4 corners were touched with the pencil (rigid fit residuals ≤ 1.4 mm), and
  `solvePnP` on the tag gives the camera pose in the robot frame, `T_base_cam = T_base_tag · T_cam_tag⁻¹`.
  The tag is re-measured at the start of every session, so a bumped tripod is detected and corrected
  (`perception/recaler_webcam.py`). Check: the points touched by the robot, projected into the webcam image with this
  pose, land on the right pixels, and so does the robot base frame:

  ![reprojection check](media/calib_reprojection.jpg)
- **Scene**: the work table is not level. A plane fitted on 9 touched points gives a 2.1° slope (residuals ≤ 2.7 mm):
  averaging the heights would have left centimetre errors at the far edge. The cable hole next to the tag is measured
  (two circles fitted on touched points) and modelled as a real hole, since MuJoCo only collides convex shapes: the
  table top is built from boxes around an octagonal opening. The tag is rendered with its real pattern, and OpenCV
  detects it in the simulated image at the same pixel as in the real one. Real and simulated views side by side:

  ![twin overlay](media/twin_overlay.jpg)
- **Wrist RealSense D435** (hand-eye calibration, `perception/calib_d435.py`): the tag corners seen from 11 robot poses
  are expressed in the hand frame and a single `solvePnP` gives the camera pose on the hand, 4.7 px mean reprojection
  error (~3–4 mm). The first method, one estimate per pose averaged, scattered by 21 mm, because the orientation of a
  small tag seen from far is noisy. Not used by the twin's control loop yet.

  ![hand-eye](media/handeye_d435.jpg)
- **Robot model**: for the same joint angles, the fingertip position computed by MuJoCo (Menagerie model, TCP 103.4 mm
  below the flange) and the one reported by the robot (`O_T_EE`) differ by **0.002 mm**. Gravity compensation is turned
  on for the simulated arm: without it, the Menagerie position actuators sag by 5–7 mm under the arm's weight, while
  the real Panda compensates gravity itself.

### Perception on the real webcam

![detection](media/detection_webcam.jpg)

`perception/detection.py`: the image is undistorted onto the ideal pinhole camera used by the twin, then HSV
thresholding finds the blue cube and the green target. The cube mask is refined with a blue-minus-red channel and
Otsu's threshold (the HSV mask missed the dark side faces), and a **3D cube model is fitted to the silhouette**: the
position and yaw whose projected cube best overlaps the mask (white outline above). The target is the intersection of
the camera ray with the sloped table plane.

Checked against the robot (open gripper centred on the cube, pencil on the target), leave-one-out cross-validation
of a small affine correction: **cube 1.8 mm (max 2.9 mm), yaw 3.2° (max 7.4°), target 2.6 mm**. The simpler
"centre of the blob + minimum rectangle" gave 2.5 mm and 9.0° (max 21.5°): seen at an angle, the side faces stretch
the blob and tilt the rectangle.

### Check in the twin, then execute

**Why joint waypoints.** The real robot must execute exactly what was checked. The expert (same logic as
`common/test_expert.py`: above the cube, down, close, up, transport, down, open, up, back) is written as 7-joint
waypoints solved by IK. libfranka joins them with its synchronized joint motion generator; the twin replays the same
joint-space interpolation. Among the four gripper orientations that grasp the cube flat (yaw modulo 90°), the planner
keeps the one with the shortest joint path, so the wrist does not spin for nothing.

**Safety checks before the robot moves**: successful simulation; cube and target unchanged since (6 mm, 8°);
robot still at the start pose used by the simulation (0.03 rad, checked on both PCs); sim/real fingertip gap < 10 mm;
joint limits, speed ≤ 0.3 and grasp force 5–40 N checked by the robot server; low libfranka collision thresholds;
failed grasp → gripper reopens and the arm goes back up.

### Results on the real robot

**First runs** (23/11/2026, speed factor 0.1, one run ≈ 30 s):

| | value |
|---|---|
| Runs executed | 8 |
| Placement error measured by the webcam (5 runs) | **1.0 / 2.9 / 3.5 / 4.1 / 2.9 mm** |
| Same runs in the twin before execution | 2.0–3.7 mm |
| Excluded, cause not identified yet | 2 runs with a final measurement 11 and 24 cm away from the target; 1 libfranka stop on the very first run |

These errors are close to the precision of the webcam measurement itself (~2–3 mm), so more runs, and a finer
measurement, are needed before drawing a statistical conclusion. The detailed log of every run (plan, simulated and measured errors, joint angles at each
waypoint, gripper width) is in `jumeau/results/`.

### Architecture

libfranka needs Linux with a 1 kHz control loop, so the work is split over two PCs: the Windows laptop runs the
webcam, the detection, MuJoCo and the planner; the Ubuntu PC wired to the robot runs a small TCP server
(`ubuntu/serveur_robot.py`, standard library only) and a C++ libfranka 0.8.0 program (`ubuntu/jumeau_robot.cpp`, the
version imposed by the robot's firmware) that reads the robot state or executes a validated list of waypoints. While
moving, the 1 kHz loop only stores the measured joint angles in atomic variables; another thread sends them every
40 ms, so the real-time loop is never slowed down by the network. (The university Wi-Fi isolates devices from each
other, so the two PCs talk through a phone hotspot.)

```
webcam ──> detection (cube, target) ──> MuJoCo twin ──> expert planned + played in the twin (E)
                                              ▲                     │ OK, nothing moved
                                              │ joint angles (25 Hz)│ waypoints (R)
                                              └──── Ubuntu: serveur_robot.py ──> jumeau_robot (libfranka) ──> Panda
```

## Obstacle avoidance: a plank on the table

The obstacle is the lab's chipboard plank, 53.6 × 10 × 3 cm. It can lie flat (3 cm high), stand on its edge (10 cm)
or stand upright (53.6 cm), anywhere on the table. The robot has to bring the cube to the target without touching it,
and should not make a detour when the plank is not in the way.

### What is learned

My first try was the same as for the pick-and-place: behavior cloning, step by step in joint space, from a planner
used as the expert (IK, detours around the plank, RRT-Connect, shortcuts). It reached 5% success, and 41% at best
with DAgger. Over ~150 steps the small errors add up, and a single one near the plank is a collision.

So I reduced what the network has to decide to three choices per episode:

- the gripper orientation (5 candidates, the cube yaw modulo 90°);
- the approach route and the transport route, each among 6: `direct`, `dessus` (go over), `cote+` / `cote-`
  (along one long side), `bout+` / `bout-` (around one end).

That makes 180 options. A classical generator does the rest, like MoveIt and the libfranka motion generator would:
IK of the key poses (it checks they are reachable), waypoints of the chosen route, smooth joint trajectory. It does
not search for a path, so if the chosen route goes through the plank, the run fails. The decision is what makes the
run succeed.

- **IL**: a classifier imitates a "discrete expert" that tries the 180 options in the twin and keeps the cheapest
  one that works (792 training scenes).
- **RL**: a contextual bandit. The robot tries one option in the twin and gets a reward (+10 minus 1 per second if
  it succeeds, −10 if it touches the plank, −5 otherwise); a network learns Q(scene, option). No expert at all,
  1800 trials.

The network outputs a ranking. On the real robot, the first options are tested in the twin (whole trajectory,
collisions), and among the safe ones the planner keeps the one that moves the arm the least.

### Results in simulation

91 evaluation scenes in the twin, the same for every method. 6–10% of random scenes have no solution at all (a
standing plank between the robot base and the cube) and are left out.

| Method | Success |
|---|---|
| Full planner (reference, ~1–30 s per plan) | 97% |
| Discrete expert (best of the 180 options) | 91% |
| IL decision, then checked in the twin | 81% → **92%** |
| RL decision, then checked in the twin | 74% → **85%** |
| IL without seeing the plank (ablation) | 62% |
| Always go straight | 58% |
| Random option | 11% |
| Step-by-step behavior cloning (DAgger) | 5% (41%) |

![decisions](avec_obstacle/results/compare_decisions.png)

Changing *what* the network learns mattered much more than the learning algorithm: from 5% to 81% with fewer
training scenes. The ablation shows the plank is really used: without it, the network almost always goes straight.
RL gets close to IL without any expert, from the reward only.

### Seeing the plank

Two cameras, which do not see the same things:

- **The webcam** (fixed, sees the whole table) fits a 3D box to the plank silhouette, for each of the three
  postures, and keeps the best one. It finds the plank anywhere, but it only has a 2D view.
- **The wrist D435**, from an observation pose above the table, gives a point cloud in the robot frame. It is
  re-registered at every photo on the tag and the table plane (about 8 mm of correction), then the clusters above
  the table give the plank footprint and height, hence its posture. It measures in 3D, but cannot measure what is
  closer than ~28 cm, like the top of the standing plank.

The two agree within 8–9 mm. The D435 is used as the reference when it sees the whole plank; the webcam covers the
rest and tells the D435 which cluster is the plank (the robot's control box on the table looks like one). The plank
is "stable" after three consistent estimates, and no plan is made while the two cameras disagree (someone moving the
plank, a hand in the view). Running the webcam fit in its own process brought an update from ~18 s to ~5 s inside
the twin, where it competed with MuJoCo, the video and the D435 for the same Python interpreter.

### Real runs

26–27/11/2026, speed factor 0.1. 7 runs went to the end (plank on its edge 3 times, standing 4 times): the cube
ended **2.4 to 8.2 mm** from the target, measured by the webcam, and no run was stopped by a collision. The log of
every run is in `jumeau/results/`.

Other attempts, and what they taught:

- 4 refused by libfranka before moving: the robot had just been moved by hand and was not back in its ready mode.
- 4 stopped by me: the chosen grasp could only be reached by turning the wrist ~200° and flipping the whole arm,
  which pulls on the D435 cable. The twin had accepted it because it was collision-free. The planner now compares
  several safe plans, keeps the one that moves the least, and refuses any plan that turns the wrist more than 150°.

Still open: with a standing plank, the webcam and the D435 do not always agree on its yaw, and planning can take
up to ~30 s when the network's first choices are not safe and the full planner takes over.

## Learning in simulation: imitation vs reinforcement learning

This part came first and runs entirely in simulation, in a **generic** MuJoCo scene (`common/panda_pick_env.py`:
plain floor, cube and target at random, cameras placed by hand), not yet in the calibrated twin.
The same task is solved twice: once by **imitating a scripted expert from camera images**, once by
**reinforcement learning from a reward only**.

| Scripted expert | Imitation learning (2 cameras) | Reinforcement learning (SAC) |
|:---:|:---:|:---:|
| ![expert](media/expert.gif) | ![il](media/il_bc_vision.gif) | ![rl](media/rl_sac.gif) |
| reads the exact simulator state | sees only a fixed camera, a wrist camera and its joint angles (right panels) | learns from a reward, no demonstrations |

### Results in simulation

Evaluated on 50 cube and target positions never seen during training.

| | Imitation learning (behavior cloning) | Reinforcement learning (SAC) |
|---|---|---|
| Input | 2 RGB cameras (96 × 96) + 7 joint angles + gripper width | exact state (tool, cube and target positions) |
| What it needs | a scripted expert and 150 demonstrations | a well-designed reward |
| Success | **50/50** | **49/50** |
| Compute (laptop CPU) | ~20 min of demos + ~50 min of training | ~4 h for 500k steps |
| Hardest part | choosing the cameras, designing an imitable expert | writing the reward |

The comparison is not yet apples to apples: the IL policy works from images, the RL policy from the exact state.
The [simulated perception](#simulated-rgb-d-perception) section replaces that exact state with positions estimated
by a simulated depth camera.

### The task

- **Robot**: Franka Panda from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), controlled with
  4 actions: a Cartesian displacement of the tool (up to 2 cm per step) and open/close the gripper.
  A differential IK (damped least squares, null-space pull for the 7th joint) turns the displacement into joint targets.
- **Episode**: 100 steps of 40 ms (4 s). Cube (4 cm) and green target mark drawn at random on the table.
- **Success**: at the end of the episode, the cube rests within 3 cm of the target **and has been released**.

### Imitation learning

1. A **scripted expert** (state machine + proportional controller) reads the exact simulator state and solves the task 49/50.
2. It records 150 demonstrations: camera images, joint angles, and its own actions.
3. A **CNN per camera + an MLP** learns to reproduce the expert's action from what the robot can actually sense.

![IL progression](media/il_progression.png)

Getting from 11/50 to 50/50 was a debugging exercise. A diagnostic script (`il/diagnose_bc_vision.py`) splits
each failure into perception, imitation and closed-loop drift, using the simulator's ground truth:

- **The expert must be imitable.** My first expert remembered its phase and used a timer to close the gripper,
  which is invisible in a single image. A memoryless expert, which decides only from the current scene, took the policy from 16/50 to 35/50.
- **Cameras matter more than the network.** With a single front camera, the gripper hid the target at the moment of placing the cube.
  Adding a wrist camera took the policy from 35/50 to 50/50: the fixed camera tells *where to go*, the wrist camera gives the final precision.
- DAgger (the expert relabels the states the policy visits) was implemented but not needed in the end.

### Reinforcement learning

SAC (Stable-Baselines3) on the exact state, with a staged reward: reach the cube, hold it, bring it to the target, release it.

![RL learning curves](media/rl_learning_curves.png)

Most of the work went into the reward, and each failure is a classic RL lesson:

- **Reward hacking (v1).** Success only asked for the cube to be on the target. The robot reached 100% by placing the cube
  and *never letting go*. Fix: success requires the gripper to be open.
- **Local optimum.** When releasing became more valuable, holding the cube far from the target still paid enough:
  the robot grasped it and stopped there. Fix: each stage must pay clearly more than the previous one.
- **Reward valley.** While opening, the gripper spent two steps neither "holding" nor "released", where the reward collapsed
  just before the success bonus. The robot never crossed it. Fix: "released" starts exactly where "holding" ends.

The final policy succeeds 49/50 and is faster than the expert. It does not retract the arm after releasing,
because nothing rewards it yet. The RL agent only does what the reward asks.

## Simulated RGB-D perception

So far the expert and the RL policy read the cube and target positions straight from the simulator.
Here they come from a **simulated depth camera** instead, through a classic detection pipeline.
Everything else the policy sees (tool position and speed, gripper opening) still comes from the robot itself,
as on the real Panda (joint encoders + kinematics).

![perception](media/perception.gif)

*Left: the simulated camera, with the detection drawn (yellow: cube, cyan: target, dot: estimated center,
white cross: true position). Right: the scene. The SAC policy (retrained, see below) receives only the estimated
positions.*

**Pipeline** (`perception/`, OpenCV + NumPy, no learning; detection + 3D take ~2 ms per image on CPU):

1. **Camera** (`realsense.py`): a fixed RGB-D camera beside the table, modelled on an Intel RealSense D435:
   424 × 240, 42° vertical field of view, depth aligned to color and rounded to 1 mm, depth noise growing with
   the square of the distance, 1% of pixels without depth, noise on the color image.
   It looks from the side because the fingers open along x: from there the cube stays visible *between* the
   fingers while it is carried (from the front, the hand hid it and tracking was lost).
2. **Detection** (`detector.py`): HSV thresholding (red cube, green target), morphological opening and closing,
   largest contour.
3. **3D position**: the pixels of the contour that have a valid depth are back-projected with the camera
   intrinsics and pose into a point cloud of the *visible surface*. The camera never sees an object's center:
   - cube (known size, upright): center = half a side below the top face and half a side behind the side face
     facing the camera;
   - target mark: middle of its extent (the median is pulled toward the camera, because the near half of a mark
     seen at an angle covers more pixels), raised by half a cube, since the policies use "where the cube's center
     should end up".
4. **Tracking**: an object that is not seen keeps its last estimate; the target, which never moves, is only
   updated when at least 90% of it is visible (the cube placed on it would otherwise shift it).

**Position error** (50 test episodes, the expert acting on the exact state, 5,000 images):

| | mean | 95th percentile | max | detected |
|---|---|---|---|---|
| Cube on the table | 0.10 cm | 0.23 cm | 0.57 cm | 100% of images |
| Cube in the gripper | 0.14 cm | 0.31 cm | 0.72 cm | 100% of images |
| Target | 0.19 cm | 0.43 cm | 0.61 cm | 95% of images |

![perception error](media/perception_error.png)

**Success with the estimated state** (the same 50 test episodes; the policies are *not* retrained):

| Policy | exact state | estimated state |
|---|---|---|
| Scripted expert | 49/50 | **50/50** |
| SAC (retrained for this test, see below) | 50/50 | **50/50** |

The SAC model files are not stored in the repository, so for this test SAC was retrained with `rl/train_rl.py`
(500k steps, same settings, seed 0). This retrained model differs from the one in the Results table above
(49/50); its own exact-state score is given in the table so that the comparison is like for like.
With millimetre-level errors, the estimated state changes nothing measurable: SAC succeeds 50/50 in both cases,
and the expert's one extra success is within the variation from episode to episode (seed 42, the one it misses with
the exact state, is a borderline case for it).

**How far does it hold?** The expert on the estimated state, in degraded conditions (`perception/robustness.py`;
SAC was not tested under these conditions):

| Condition | cube error (mean / max) | success |
|---|---|---|
| Reference (noisy camera, perfect calibration) | 0.11 / 0.54 cm | 50/50 |
| Rendering with shadows | 0.11 / 0.62 cm | 48/50 |
| Camera pose wrong by 5 mm and 1° | 1.19 / 2.22 cm | 48/50 |
| Camera pose wrong by 10 mm and 2° | 2.34 / 3.61 cm | **0/50** |

The detection itself is not the weak point; **the camera calibration is**. With a 10 mm / 2° error on the camera
pose, the cube is estimated about 2 cm away from where it really is (18 mm sideways and 12 mm too low in the
episode I inspected). The gripper, which knows its own position exactly, goes down beside the cube and closes on
nothing, every time. On the real robot this means a careful hand-eye calibration, or a final approach guided by
the wrist camera.

**Limits.** This is simulation only, and the setting is easy for a camera: two saturated colors on a plain floor,
constant lighting, no motion blur, no reflections, image and depth perfectly synchronized, exact intrinsics, a
single cube of known size that stays upright. Shadows are off by default (much faster with software rendering) and
tested separately above. A real RealSense adds edge artifacts and holes on dark or shiny surfaces, and the hand-eye
calibration is never perfect, so the millimetre errors above are a lower bound, not a prediction for the real setup.
The GIF shows the noise-free image of the same camera for readability; the detection drawn on it ran on the noisy
image, as in all the numbers above.

## Repository layout

```
common/   panda_pick_env.py   Gymnasium env: robot, cube, target, 2 cameras, IK, reward
          test_expert.py      scripted, memoryless expert + its test
          twin_scene.py       digital twin scene: the real tables, hole, tag, cube, target, calibrated cameras
il/       record_demos*.py, train_bc*.py, diagnose_bc_vision.py, dagger.py
rl/       train_rl.py, eval_rl.py
perception/
          realsense.py        simulated RGB-D camera (RealSense D435-like)
          detector.py         HSV detection, 3D position of cube and target, tracking
          eval_perception.py  position error on 50 episodes + figure
          compare_policies.py success with the exact vs the estimated state (expert, SAC)
          robustness.py       shadows and camera calibration errors
          make_gif.py         perception GIF / MP4
          results/            JSON and CSV of the numbers above
          real setup (webcam + RealSense D435, robot base frame):
          calib_webcam.py, apriltag_pose.py, recaler_webcam.py   intrinsics, tag pose, per-session re-calibration
          detection.py        cube (3D model fit) and target on the real webcam
          evaluer_detection.py, corriger_detection.py, erreur_detection.py, verifier_lacet.py   error vs the robot
          d435_tag.py, calib_d435.py   hand-eye calibration of the wrist D435
          detection_planche.py, recalage_d435.py   plank from the webcam and the D435, D435 re-registration
          mesure_crayon.py, evaluer_crayon.py      guided pencil measurements (grips the pencil first) and error
          superposition.py    real image vs rendered twin
          *.json              calibration of the lab setup
avec_obstacle/
          common/             plank (postures, placement), twin env with the plank, planner expert,
                              decisions (routes, generator, evaluation), common evaluation
          il/                 decision IL (+ ablation), step-by-step BC and DAgger, comparisons
          rl/                 decision RL (contextual bandit), SAC with demonstrations
          videos/             video_planche.py
          results/            comparison tables and figures
jumeau/   jumeau_live.py      live twin: E = simulate, R = execute, V = record real | twin video
          planif_planche.py   with the plank: ranked decisions checked in the twin, least wrist rotation
          suivi_planche.py    plank tracking (webcam fit in its own process, D435, fusion)
          planif_expert.py    expert as joint waypoints, replay and checks in MuJoCo
          lien_robot.py       client for the Ubuntu server
          video_sim.py        clean video / GIF of a simulated episode
          results/            log of the real runs
ubuntu/   serveur_robot.py    TCP server on the robot PC
          jumeau_robot.cpp    libfranka 0.8.0 program: read the state, execute waypoints, stream joint angles
          envoyer_ubuntu.py   copies these files to the robot PC over the network
videos/   make_videos.py      renders the videos above
media/    GIFs, MP4s and figures used in this README
```

## Quick start

```bash
python -m venv venv && venv\Scripts\activate          # Windows (source venv/bin/activate on Linux)
python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git   # robot model, at the repo root

python common/test_expert.py --view       # watch the expert
python il/record_demos_vision.py          # 150 demos with both cameras
python il/train_bc_vision.py              # train + evaluate on 50 unseen episodes
python il/train_bc_vision.py --eval --view
python rl/train_rl.py 500000              # SAC, a few hours on CPU
python rl/eval_rl.py --view
python videos/make_videos.py              # expert / IL / RL videos

python perception/eval_perception.py      # position error on 50 episodes (~10 min on CPU)
python perception/compare_policies.py     # exact vs estimated state (expert, + SAC if rl/sac_best.zip exists)
python perception/robustness.py           # shadows, calibration errors (~1 h on CPU)
python perception/make_gif.py             # perception GIF
```

Every script writes its outputs (demos, models, logs) next to itself.

**Digital twin with the real robot** (the calibration files in `perception/` describe my lab setup):

```bash
# Ubuntu PC connected to the robot: copy jumeau_robot.cpp into libfranka-0.8.0/examples, add it to the examples'
# CMakeLists.txt, build it (cmake --build . --target jumeau_robot), then
python3 serveur_robot.py --confirmer       # --confirmer: press Enter on this PC before every motion

# Windows PC (same network)
python perception/recaler_webcam.py       # tag visible: update the webcam pose
python jumeau/jumeau_live.py --robot <IP of the Ubuntu PC>
python jumeau/jumeau_live.py              # without robot: simulation only
python jumeau/video_sim.py --gif          # video of the last trajectory, simulated
python jumeau/jumeau_live.py --robot <IP> --planche --d435   # with the plank (O = observation pose)
```

**Obstacle part, in simulation** (models are not stored, train them first):

```bash
python avec_obstacle/il/record_decisions_planche.py 3000     # every option evaluated per scene (parallel)
python avec_obstacle/il/train_decision_il.py                 # IL, then --sans-planche for the ablation
python avec_obstacle/rl/train_decision_rl.py 30 --essais 200 # RL, no expert
python avec_obstacle/il/compare_decisions.py                 # table + figure
python avec_obstacle/videos/video_planche.py --politique decision_il --mode jumeau
```

## Next steps

- A longer series of real runs (10–20 positions, yaw 0–90°) to compare the twin and reality statistically.
- Train the IL and RL policies in the **calibrated twin** scene, then run them on the real robot through the same
  check-then-execute loop, against the expert.
- Use the wrist RealSense D435 (already hand-eye calibrated): check the grasp, refine the cube position just before
  grasping, see obstacles in depth.
- Use the D435 depth for a full 3D model of the scene, not only known objects on a plane, and keep the twin in
  sync with it automatically.
- Re-measure the cube with the wrist camera just before grasping.
- Human demonstrations through teleoperation, then Isaac Sim.
