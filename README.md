# Optical Flow & Visual Potential Field Navigation

A three-part computer vision project exploring optical flow, from real-time sparse feature
tracking to a fully closed-loop, camera-only autonomous driving simulation.

| Part | File | Description |
|------|------|-------------|
| 1. Sparse Optical Flow | `AbhirupPaul_Task1_S1.py` | Real-time pyramidal Lucas-Kanade tracking with forward-backward consistency checks |
| Bonus. Dense Optical Flow | `AbhirupPaul_Task1_Bonus.py` | Sparse-to-dense flow via Delaunay triangulation + HSV motion visualization |
| 2. Visual Navigation | `AbhirupPaul_Task1_S2.py` | PyBullet simulation of camera-only autonomous driving using a Visual Potential Field and Sliding Mode Control |
| Setup | `simulation_setup.py` | Builds the PyBullet road, slalom obstacles, and racecar |

---

## Part 1 — Sparse Optical Flow (`AbhirupPaul_Task1_S1.py`)

Tracks a sparse set of strong corner features (Shi-Tomasi) frame-to-frame using pyramidal
Lucas-Kanade optical flow.

- **Forward-backward consistency check**: each point is tracked forward then backward; if
  it doesn't return close to its original position, the track is discarded as unreliable.
- **Trajectory trails**: each surviving point keeps a rolling history of positions, drawn
  as a polyline.
- **Periodic re-detection**: every 5 frames, new corners are detected in regions not
  already covered by an active track (using a mask).
- Live FPS counter and track-count overlay.

## Bonus — Dense Optical Flow (`AbhirupPaul_Task1_Bonus.py`)

Extends sparse tracking to a dense, per-pixel flow field.

- Tracks up to 500 loosely-qualified corners with pyramidal LK.
- **`sparse_to_dense()`**: interpolates the sparse flow vectors over every pixel using
  Delaunay triangulation (`scipy.interpolate.LinearNDInterpolator`) — far better than
  blurring a sparse flow image, since it doesn't dilute real flow values with zeros.
- **`draw_flow()`**: samples the dense field on a grid and draws motion arrows.
- **`draw_hsv()`**: visualizes the entire dense flow field as a color image — hue encodes
  direction, brightness encodes speed.

## Part 2 — Visual Potential Field Navigation (`AbhirupPaul_Task1_S2.py`)

A complete, map-free autonomous driving pipeline running inside a PyBullet simulation
(road + 5 slalom obstacles + racecar, built by `simulation_setup.py`). The car drives
using **only its forward-facing monocular camera**.

### Perception
- **Optical flow**: pyramidal LK tracks up to 300 points per frame.
- **Focus of Expansion (FOE)**: estimated via least-squares from the flow field —
  represents the car's current direction of travel.
- **Time-to-Contact (TTC)**: per-point TTC = distance from FOE ÷ flow magnitude.
- **Obstacle map**: built from optical-flow *residuals* — points whose flow deviates from
  the pure ego-motion (radial) pattern are flagged as obstacles, isolated via Otsu
  thresholding, then Gaussian-blurred and Sobel-differentiated to get a repulsive gradient
  field.
- **Raycasting**: a 5-ray fan ahead of the car gives ground-truth obstacle distances, used
  for threat classification (`clear` / `warn` / `brake` / `emergency`) with hysteresis to
  prevent flickering.

### Planning & Control
- **Visual Potential Field (VPF)**: combines an attractive force (toward the goal/lane
  centre), an optical-flow-derived obstacle repulsion, and a Morse-potential road-boundary
  repulsion into a single resultant force vector.
- **Lane centring**: PD controller keeping the car at the road centreline.
- **Dodge maneuver state machine**: locks onto an obstacle, commits to a lateral dodge
  target, and releases once the car has cleared it, followed by a re-centring cooldown.
- **GTSMC (Sliding Mode Control)**: bang-bang sliding-mode controllers for both lateral
  steering (heading error) and longitudinal speed.
- **Failsafes**: emergency override at very close range, and a hard boundary guard if the
  car nears the road edge.

### Visualization
Live HUD overlay showing optical flow vectors, FOE, raycast hit bars, obstacle mask,
threat level, speed, steering angle, and current driving mode (`CENTRE` / `DODGE` /
`RECENTRE` / `BOUNDARY!`).

---

## Requirements

```
opencv-python
numpy
scipy
pybullet
```

## Running

```bash
# Part 1 — sparse optical flow on the provided video
python AbhirupPaul_Task1_S1.py

# Bonus — dense optical flow on the provided video
python AbhirupPaul_Task1_Bonus.py

# Part 2 — full visual navigation simulation
python AbhirupPaul_Task1_S2.py
```

`AbhirupPaul_Task1_S1.py` and `AbhirupPaul_Task1_Bonus.py` expect `OPTICAL_FLOW.mp4` in the
working directory. `AbhirupPaul_Task1_S2.py` requires no input video — it renders camera
frames directly from the PyBullet simulation.
