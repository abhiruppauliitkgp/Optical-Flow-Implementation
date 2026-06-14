"""
vpf_navigation.py
=================
Visual Potential Field (VPF) based autonomous navigation.
Uses a monocular camera + optical flow to navigate a car in PyBullet simulation.
The car avoids obstacles and stays on the road without any pre-loaded map.
"""

import cv2
import numpy as np
import pybullet as p
import time
from simulation_setup import setup_simulation


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 640, 480   # camera image resolution in pixels
FOV           = 90         # field of view in degrees (how wide the camera sees)


# ─────────────────────────────────────────────────────────────────────────────
# ROAD GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

ROAD_HALF_W = 1.16   # half the road width in metres (total road = 2.32m)
CAR_HALF_W  = 0.25   # half the car width in metres


# ─────────────────────────────────────────────────────────────────────────────
# SPEED SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

SPEED_REF = 10.0    # normal cruising speed in m/s
STEER_MAX = 0.6     # maximum steering angle in radians


# ─────────────────────────────────────────────────────────────────────────────
# LANE CENTRING CONTROLLER GAINS
# ─────────────────────────────────────────────────────────────────────────────

CENTRE_KP = 8.0   # proportional gain — how strongly to correct lateral position
CENTRE_KD = 2.0   # derivative gain — damps oscillation by reacting to rate of change


# ─────────────────────────────────────────────────────────────────────────────
# RAYCAST THREAT THRESHOLDS (distances in metres)
# ─────────────────────────────────────────────────────────────────────────────

DIST_WARN  = 12.0   # obstacle detected — begin monitoring
DIST_BRAKE =  7.0   # obstacle close — start slowing down
DIST_EMERG =  3.5   # obstacle very close — emergency action needed

# Target speeds for each threat level
OBS_SPEED_WARN      = 6.0   # slow to 6 m/s when a warning is active
OBS_SPEED_BRAKE     = 3.5   # slow to 3.5 m/s when braking
OBS_SPEED_EMERGENCY = 1.5   # slow to 1.5 m/s in emergency

# Ray fan configuration
RAY_ANGLES_DEG = [-25, -10, 0, 10, 25]  # five rays spread across a 50° arc ahead
RAY_LENGTH     = 15.0                    # each ray reaches 15 metres ahead
RAY_HEIGHT     = 0.35                    # rays cast from 0.35m above ground


# ─────────────────────────────────────────────────────────────────────────────
# THREAT HYSTERESIS
# ─────────────────────────────────────────────────────────────────────────────

# Number of consecutive frames at a lower threat before de-escalating
# Prevents the car from relaxing too soon after an obstacle briefly disappears
THREAT_DEESCALATE_FRAMES = 6


# ─────────────────────────────────────────────────────────────────────────────
# DODGE PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

DODGE_Y_TARGET      = 0.55   # how far sideways (metres) to move when dodging
DODGE_RELEASE_MARGIN = 2.5   # how far past the obstacle before releasing the dodge lock


# ─────────────────────────────────────────────────────────────────────────────
# POST-DODGE RE-CENTRING
# ─────────────────────────────────────────────────────────────────────────────

RECENTER_COOLDOWN = 1.0    # seconds to re-centre after a dodge before accepting new threats
RECENTER_BOOST    = 4.0    # multiplier on centring force during re-centring
RECENTER_FY_DECAY = 0.20   # how fast the dodge force fades out (20% per step)


# ─────────────────────────────────────────────────────────────────────────────
# HARD BOUNDARY GUARD (last resort if everything else fails)
# ─────────────────────────────────────────────────────────────────────────────

BOUNDARY_Y     = 0.72   # if car_y exceeds this, activate emergency boundary steering
BOUNDARY_STEER = 0.6    # steering strength to push back from the boundary
BOUNDARY_SPEED = 3.0    # maximum allowed speed when boundary is active


# ─────────────────────────────────────────────────────────────────────────────
# VPF / APF PARAMETERS (from the paper)
# ─────────────────────────────────────────────────────────────────────────────

APF_A           = 0.5     # height (maximum value) of the road potential field walls
APF_B           = 1.0     # steepness of the road field walls
APF_C2_STRAIGHT = 0.005   # curvature parameter for straight road
APF_C2_CURVED   = 5e-6    # curvature parameter for curved road
APF_C1          = 0.0     # linear term (set to zero — not needed)
APF_C0R         =  0.91   # right lane boundary offset from centre (metres)
APF_C0L         = -0.91   # left lane boundary offset from centre (metres)
DELTA_X         = 1.0e-10 # tiny offset to prevent division by zero
ALPHA_ATT       = 0.5     # attractive force strength (pulls car toward goal)
GAMMA_REP       = 1.5     # obstacle repulsion strength
LAMBDA_X        = 0.4     # weight of road field in X direction
LAMBDA_Y        = 0.4     # weight of road field in Y direction
SIGMA_GAUSS     = WIDTH / 2.0  # Gaussian blur width for obstacle map smoothing


# ─────────────────────────────────────────────────────────────────────────────
# GTSMC CONTROLLER GAINS
# ─────────────────────────────────────────────────────────────────────────────

CR = 2.0   # lateral sliding manifold gain (balances heading error vs rate)
U0 = 0.6   # maximum steering rate (how fast the steering angle changes)
CL = 1.5   # longitudinal sliding manifold gain
A0 = 3.0   # maximum acceleration/deceleration rate


# ─────────────────────────────────────────────────────────────────────────────
# OPTICAL FLOW SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

lk_params = dict(
    winSize=(25, 25),    # search window — larger than sparse flow for more accuracy
    maxLevel=3,          # 4 pyramid levels for handling different motion speeds
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    # stop after 20 iterations OR when improvement < 0.03
)

feature_params = dict(
    maxCorners=300,      # track up to 300 points for good FOE estimation
    qualityLevel=0.01,   # low threshold — accept most corners for wide coverage
    minDistance=8,       # corners must be at least 8 pixels apart
    blockSize=7
)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE STATE (persistent variables that carry over between frames)
# ─────────────────────────────────────────────────────────────────────────────

prev_gray  = None         # previous frame in grayscale
prev_pts   = None         # previously tracked corner points
delta_f    = 0.0          # current steering angle (GTSMC integrator state)
speed_cur  = SPEED_REF    # current speed target

_prev_y          = 0.0    # car's y-position in the previous frame (for derivative)
_smooth_dodge_FY = 0.0    # smoothed lateral dodge force
_cooldown_timer  = 0.0    # countdown after a dodge before accepting new obstacles
_last_threat     = "clear" # threat level in the previous frame

# Hysteresis state — tracks candidate vs confirmed threat level
_candidate_threat = "clear"
_candidate_frames = 0
_confirmed_threat = "clear"

# Per-obstacle dodge lock state
_dodge_obs_x    = -999.0  # estimated world-x position of the current obstacle
_dodge_target_y =  0.0    # target y-position for the current dodge
_dodge_active   = False   # whether a dodge is currently in progress

_last_min_dist  = RAY_LENGTH  # most recent minimum raycast distance


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA
# ═════════════════════════════════════════════════════════════════════════════

def get_camera_image(car_id):
    """
    Renders a camera image from the car's windshield perspective.
    Returns both a grayscale version (for optical flow) and RGB (for display).
    """
    # Get the car's current position and orientation in the world
    pos, orn = p.getBasePositionAndOrientation(car_id)

    # Convert the orientation quaternion to a 3x3 rotation matrix
    rot = p.getMatrixFromQuaternion(orn)

    # Extract the car's forward direction — first column of the rotation matrix
    # rot[0], rot[3], rot[6] are the x, y, z components of the forward vector
    forward = np.array([rot[0], rot[3], rot[6]])

    # Place the camera 1.05m ahead of the car and 0.32m up (windshield position)
    cam_pos = np.array(pos) + forward * 1.05 + np.array([0, 0, 0.32])

    # The camera looks 5m ahead and slightly downward (toward the road)
    target  = cam_pos + forward * 5 + np.array([0, 0, -0.25])

    # Compute the view matrix (camera position and orientation)
    view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])

    # Compute the projection matrix (FOV, aspect ratio, near/far clip planes)
    proj = p.computeProjectionMatrixFOV(FOV, WIDTH / HEIGHT, 0.1, 50)

    # Render the image — returns width, height, RGB, depth, segmentation
    _, _, rgb, _, _ = p.getCameraImage(WIDTH, HEIGHT, view, proj)

    # Convert raw pixel data to a numpy array and drop the alpha channel
    frame_rgb  = np.array(rgb, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))[:, :, :3]

    # Convert RGB to grayscale for optical flow processing
    frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    return frame_gray, frame_rgb


# ═════════════════════════════════════════════════════════════════════════════
# RAYCASTING — detects obstacles ahead of the car
# ═════════════════════════════════════════════════════════════════════════════

def raycast_obstacle(car_id):
    """
    Casts five rays in a fan pattern ahead of the car.
    Returns the distance and world position of the closest obstacle hit,
    plus per-ray details for visualisation.
    """
    # Get car's position and orientation
    pos, orn  = p.getBasePositionAndOrientation(car_id)
    rot       = p.getMatrixFromQuaternion(orn)

    # Car's forward and left direction vectors in world space
    fwd       = np.array([rot[0], rot[3], rot[6]])
    left_vec  = np.array([rot[1], rot[4], rot[7]])

    # Start all rays from just ahead of and slightly above the car
    ray_start = np.array(pos) + fwd * 0.6 + np.array([0, 0, RAY_HEIGHT])

    hits       = []
    best_dist  = RAY_LENGTH        # start assuming no obstacle (max range)
    best_hit_y = float(pos[1])     # default to car's own y position

    for ang_deg in RAY_ANGLES_DEG:
        ang = np.radians(ang_deg)  # convert angle to radians

        # Compute ray direction: blend forward and left vectors by the angle
        # ang=0 → straight ahead, positive → left, negative → right
        d   = np.cos(ang) * fwd + np.sin(ang) * left_vec

        # End point of this ray
        end = ray_start + d * RAY_LENGTH

        # Cast the ray — returns hit object ID, fraction, normal, position
        res = p.rayTest(ray_start.tolist(), end.tolist())[0]
        hid = res[0]  # ID of the object hit (0 = nothing)

        if hid > 0 and hid != car_id:
            # Something was hit and it's not the car itself
            hit_pos  = np.array(res[3])   # world-space position of the hit point
            hit_dist = float(np.linalg.norm(hit_pos - ray_start))  # distance to hit

            # Update best (closest) hit
            if hit_dist < best_dist:
                best_dist  = hit_dist
                best_hit_y = float(hit_pos[1])  # world-Y tells us which side the obstacle is on
        else:
            hit_dist = RAY_LENGTH  # no hit — ray reached maximum range

        hits.append((hit_dist, ang_deg, hid > 0 and hid != car_id))

    # Compute average distances for left-angled and right-angled rays
    all_dists   = [h[0] for h in hits]
    min_dist    = min(all_dists)
    left_dists  = [h[0] for h in hits if h[1] > 0]   # positive angles = left
    right_dists = [h[0] for h in hits if h[1] < 0]   # negative angles = right
    centre_d    = next((h[0] for h in hits if h[1] == 0), RAY_LENGTH)

    dist_left  = float(np.mean(left_dists))  if left_dists  else RAY_LENGTH
    dist_right = float(np.mean(right_dists)) if right_dists else RAY_LENGTH

    # Assign the centre ray's distance to whichever side is closer
    if dist_left < dist_right:
        dist_left  = min(dist_left,  centre_d)
    else:
        dist_right = min(dist_right, centre_d)

    return min_dist, best_hit_y, dist_left, dist_right, hits


def classify_raw_threat(dist):
    """Converts a distance into a threat level string."""
    if dist > DIST_WARN:    return "clear"
    elif dist > DIST_BRAKE: return "warn"
    elif dist > DIST_EMERG: return "brake"
    else:                   return "emergency"


# Numerical rank for each threat level (used for comparisons)
THREAT_RANK = {"clear": 0, "warn": 1, "brake": 2, "emergency": 3}

def apply_hysteresis(raw_threat):
    """
    Smooths out noisy threat level changes.
    - Escalation (danger increasing) happens instantly — safety first
    - De-escalation (danger decreasing) requires 6 consecutive frames at the lower level
      to prevent the car from relaxing due to a single noisy ray reading
    """
    global _candidate_threat, _candidate_frames, _confirmed_threat

    raw_rank  = THREAT_RANK.get(raw_threat, 0)
    conf_rank = THREAT_RANK.get(_confirmed_threat, 0)

    if raw_rank >= conf_rank:
        # New threat is equal or higher — accept immediately
        _confirmed_threat = raw_threat
        _candidate_threat = raw_threat
        _candidate_frames = THREAT_DEESCALATE_FRAMES
    else:
        # New threat is lower — count frames before accepting the de-escalation
        if raw_threat == _candidate_threat:
            _candidate_frames += 1
        else:
            _candidate_threat = raw_threat
            _candidate_frames = 1

        if _candidate_frames >= THREAT_DEESCALATE_FRAMES:
            _confirmed_threat = _candidate_threat  # enough frames — de-escalate

    return _confirmed_threat


# ═════════════════════════════════════════════════════════════════════════════
# OPTICAL FLOW + FOE
# ═════════════════════════════════════════════════════════════════════════════

def compute_optical_flow(prev_g, curr_g, prev_p):
    """
    Tracks feature points from the previous frame to the current frame.
    Returns old positions, new positions, flow vectors, and updated point list.
    """
    # If we don't have enough points, detect fresh corners
    if prev_p is None or len(prev_p) < 30:
        prev_p = cv2.goodFeaturesToTrack(prev_g, mask=None, **feature_params)
        if prev_p is None:
            return None, None, None, None  # no features found — return empty

    # Track points from previous frame to current frame using pyramidal LK
    next_p, status, _ = cv2.calcOpticalFlowPyrLK(prev_g, curr_g, prev_p, None, **lk_params)

    if next_p is None or status is None:
        return None, None, None, None

    # Keep only successfully tracked points (status == 1)
    mask     = status.ravel() == 1
    good_old = prev_p[mask]   # positions in previous frame
    good_new = next_p[mask]   # positions in current frame

    if len(good_new) == 0:
        return None, None, None, None

    # flow = new_position - old_position (how far each point moved)
    return good_old, good_new, good_new - good_old, good_new.reshape(-1, 1, 2)


def compute_foe(good_old, flow_vecs):
    """
    Computes the Focus of Expansion (FOE) — the point in the image that all
    flow vectors radiate away from. Represents the car's direction of travel.

    Uses least squares: FOE = (AᵀA)⁻¹Aᵀb
    Each flow vector contributes one row to matrices A and b.
    """
    if good_old is None or len(good_old) < 4:
        # Not enough points — return image centre as default
        return WIDTH / 2.0, HEIGHT / 2.0

    pts = good_old.reshape(-1, 2)   # pixel positions: shape (N, 2)
    vxy = flow_vecs.reshape(-1, 2)  # flow vectors: shape (N, 2)

    # Build matrix A: each row is [vy, -vx] for that pixel
    A = np.column_stack([vxy[:, 1], -vxy[:, 0]])

    # Build vector b: each entry is x*vy - y*vx for that pixel
    b = pts[:, 0] * vxy[:, 1] - pts[:, 1] * vxy[:, 0]

    try:
        ATA = A.T @ A  # 2x2 matrix

        # Check if ATA is near-singular (happens when car barely moves)
        det = ATA[0,0]*ATA[1,1] - ATA[0,1]**2
        if abs(det) < 1e-8:
            return WIDTH/2.0, HEIGHT/2.0  # degenerate case — return centre

        # Solve the least squares system: ATA * FOE = A^T * b
        foe = np.linalg.solve(ATA, A.T @ b)

        # Clip FOE to stay within image boundaries
        return float(np.clip(foe[0], 0, WIDTH)), float(np.clip(foe[1], 0, HEIGHT))

    except np.linalg.LinAlgError:
        return WIDTH / 2.0, HEIGHT / 2.0  # fallback on numerical failure


# ═════════════════════════════════════════════════════════════════════════════
# CENTRING FORCE — keeps the car in the middle of its lane
# ═════════════════════════════════════════════════════════════════════════════

def compute_centre_force(car_id, dt, boost=1.0):
    """
    Computes a lateral force to keep the car at y = 0 (road centre).
    Uses a PD controller:
      - Proportional term: pushes back proportional to how far off-centre
      - Derivative term: damps oscillation by reacting to how fast y is changing
    The boost multiplier increases centring force during post-dodge re-centring.
    """
    global _prev_y

    pos, _ = p.getBasePositionAndOrientation(car_id)
    y      = float(pos[1])  # current lateral position

    # Estimate lateral velocity by finite difference
    dy_dt  = (y - _prev_y) / max(dt, 1e-4)

    _prev_y = y  # store for next frame

    # PD control: negative because we want to push toward y=0
    # If y > 0 (car is left of centre), force pushes right (negative)
    force = float(-(y * CENTRE_KP + dy_dt * CENTRE_KD) * boost)

    return force, abs(y)  # also return |y| for boundary checking


# ═════════════════════════════════════════════════════════════════════════════
# DODGE FORCE — steers the car around obstacles
# ═════════════════════════════════════════════════════════════════════════════

def compute_dodge_force(threat, hit_world_y, car_x, car_y, min_dist, dt):
    """
    Decides if and how to dodge an obstacle.
    Key design decisions:
    - Direction: based on world-Y of the hit point (absolute, not relative)
    - Commitment: once a dodge is started, it's locked until the car passes the obstacle
    - Magnitude: fixed step (not distance-scaled) to ensure the obstacle is fully cleared
    """
    global _smooth_dodge_FY, _cooldown_timer, _last_threat
    global _dodge_obs_x, _dodge_target_y, _dodge_active
    global delta_f

    # Set speed target based on threat level
    if threat == "clear":
        obs_speed = SPEED_REF
    elif threat == "warn":
        obs_speed = OBS_SPEED_WARN
    elif threat == "brake":
        obs_speed = OBS_SPEED_BRAKE
    else:
        obs_speed = OBS_SPEED_EMERGENCY

    # ── COOLDOWN TIMER MANAGEMENT ─────────────────────────────────────────────

    # Start cooldown when threat clears (car just passed an obstacle)
    if _last_threat != "clear" and threat == "clear":
        _cooldown_timer = RECENTER_COOLDOWN
    elif threat != "clear":
        _cooldown_timer = 0.0   # reset cooldown if a new threat appears

    _last_threat = threat

    # Count down the cooldown timer
    in_cooldown = (_cooldown_timer > 0.0) and (threat == "clear")
    if in_cooldown:
        _cooldown_timer = max(0.0, _cooldown_timer - dt)

    # If a new obstacle arrives during cooldown, cancel the cooldown immediately
    # Without this, the car would ignore the new obstacle while re-centring
    if threat != "clear" and in_cooldown:
        _cooldown_timer  = 0.0
        in_cooldown      = False
        _smooth_dodge_FY = 0.0  # discard the decaying force from the previous dodge

    # ── RELEASE EXISTING DODGE LOCK ───────────────────────────────────────────

    # Once the car has passed the obstacle (with a safety margin), release the lock
    if _dodge_active and car_x > _dodge_obs_x + DODGE_RELEASE_MARGIN:
        _dodge_active   = False
        _dodge_obs_x    = -999.0
        _dodge_target_y = 0.0
        _cooldown_timer = RECENTER_COOLDOWN  # begin re-centring phase

    # ── LOCK ONTO A NEW OBSTACLE ──────────────────────────────────────────────

    if threat != "clear" and not _dodge_active and not in_cooldown:

        # Determine dodge direction from the obstacle's world-Y coordinate
        # This is absolute — it doesn't matter where the car currently is
        if abs(hit_world_y) < 0.08:
            # Obstacle is almost perfectly centred — default to dodging right
            _dodge_target_y = -DODGE_Y_TARGET
        elif hit_world_y > 0:
            # Obstacle is on the LEFT side of the road → dodge RIGHT (negative y)
            _dodge_target_y = -DODGE_Y_TARGET
        else:
            # Obstacle is on the RIGHT side of the road → dodge LEFT (positive y)
            _dodge_target_y =  DODGE_Y_TARGET

        # Estimate the obstacle's world-x position using the raycast distance
        _dodge_obs_x  = car_x + min_dist
        _dodge_active = True

        # Reset the steering integrator so previous wind-up doesn't affect this dodge
        delta_f = 0.0

        print(f"[DODGE] hit_y={hit_world_y:+.2f} → target_y={_dodge_target_y:+.2f}"
              f"  car_y={car_y:+.2f}  obs_x≈{_dodge_obs_x:.1f}"
              f"  dist={min_dist:.1f}m")

    # ── COMPUTE THE LATERAL DODGE FORCE ──────────────────────────────────────

    if _dodge_active and not in_cooldown:
        # Proportional control toward the target y position — fixed commitment
        dodge_FY = 12.0 * (_dodge_target_y - car_y)
        _smooth_dodge_FY = dodge_FY

    elif in_cooldown:
        # Gradually decay the force back to zero during re-centring
        _smooth_dodge_FY *= (1.0 - RECENTER_FY_DECAY)
        if abs(_smooth_dodge_FY) < 0.01:
            _smooth_dodge_FY = 0.0   # snap to zero once negligibly small

    else:
        # No dodge active, no cooldown — no lateral force
        _smooth_dodge_FY = 0.0

    centre_boost     = RECENTER_BOOST if in_cooldown else 1.0
    bleed_integrator = in_cooldown  # flag to slowly bleed the steering integrator during re-centring

    return float(_smooth_dodge_FY), float(obs_speed), in_cooldown, centre_boost, bleed_integrator


# ═════════════════════════════════════════════════════════════════════════════
# VPF FORCES (Visual Potential Field — from the paper)
# ═════════════════════════════════════════════════════════════════════════════

def build_obstacle_map(good_new, flow_vecs, xFOE, yFOE):
    """
    Builds a binary obstacle map from optical flow disturbances.

    Background pixels flow directly away from the FOE (pure ego-motion).
    Obstacle pixels have disturbed flow that deviates from this pattern.
    We measure the deviation as a residual and threshold it to find obstacles.

    Then we Gaussian-smooth the binary map and take its gradient —
    the gradient direction points away from each obstacle.
    """
    O = np.zeros((HEIGHT, WIDTH), dtype=np.float32)  # blank obstacle map

    if good_new is not None and len(good_new) > 0:
        pts = good_new.reshape(-1, 2)   # current positions of tracked points
        vxy = flow_vecs.reshape(-1, 2)  # flow vectors at each tracked point

        # Distance of each point from the FOE
        df = np.sqrt((pts[:,0]-xFOE)**2 + (pts[:,1]-yFOE)**2) + 1e-6

        # Magnitude of each flow vector
        fm = np.sqrt(vxy[:,0]**2 + vxy[:,1]**2) + 1e-6

        # Dot product of the flow vector with the unit vector pointing away from FOE
        # For background pixels, this should equal fm (flow points away from FOE)
        dot = vxy[:,0]*(pts[:,0]-xFOE)/df + vxy[:,1]*(pts[:,1]-yFOE)/df

        # Residual = how much the flow DOESN'T point away from the FOE
        # High residual → obstacle pixel. Low residual → background pixel.
        res = fm - np.abs(dot)

        # Paint each residual value into the obstacle map at the tracked point's location
        xs = np.clip(pts[:,0].astype(int), 0, WIDTH-1)
        ys = np.clip(pts[:,1].astype(int), 0, HEIGHT-1)
        for i in range(len(xs)):
            O[ys[i], xs[i]] = max(O[ys[i], xs[i]], float(res[i]))

    # Normalise the obstacle map to 0-255 range for Otsu thresholding
    O_u8 = cv2.normalize(O, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu thresholding: automatically finds the best threshold to separate
    # obstacle pixels (high residual) from background pixels (low residual)
    if O_u8.max() > 0:
        _, mask = cv2.threshold(O_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = O_u8  # all zeros — no obstacles detected

    # Gaussian blur smooths the binary mask into a continuous hill shape
    # Large sigma (half image width) creates a broad, smooth potential field
    k  = min(int(6*SIGMA_GAUSS)|1, min(WIDTH,HEIGHT)-1)|1  # kernel size (must be odd)
    bl = cv2.GaussianBlur(mask.astype(np.float32), (k, k), SIGMA_GAUSS)

    # Sobel gradient in x and y — gives the direction to push AWAY from obstacles
    return mask, cv2.Sobel(bl, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(bl, cv2.CV_32F, 0, 1, ksize=3)


def compute_road_potential_field(car_id, xFOE):
    """
    Computes the road boundary repulsive force using a modified Morse potential.
    The potential is zero at the lane centre and rises steeply near the boundaries.
    The gradient (force) is computed numerically using finite differences.

    FOE position determines whether we use straight or curved road parameters.
    """
    pos, _ = p.getBasePositionAndOrientation(car_id)
    x_veh  = float(pos[0])   # car's world x position
    y_veh  = float(pos[1])   # car's world y position

    # Normalise FOE x to [0, 1] range and decide road curvature
    foe_n = xFOE / WIDTH
    if 0.35 < foe_n < 0.65:
        c2 = APF_C2_STRAIGHT  # FOE near centre → road is straight
    elif foe_n <= 0.35:
        c2 = -APF_C2_CURVED   # FOE left → left curve
    else:
        c2 =  APF_C2_CURVED   # FOE right → right curve

    # Compute lane boundary y-positions as a function of x
    xd  = x_veh + DELTA_X        # tiny offset prevents division by zero
    yr  = c2*xd**2 + APF_C0R     # right boundary position
    yl  = c2*xd**2 + APF_C0L     # left boundary position

    # Slope of the line perpendicular to the lane centre
    denom = 2*c2*xd + APF_C1
    m     = -1.0 / (denom if abs(denom) > 1e-9 else 1e-9)

    # Y-intercepts of the perpendicular lines at the boundaries
    by_r = yr - m*xd
    by_l = yl - m*xd

    eps = 1e-4  # small step for numerical gradient

    def Utotal(y_):
        """Morse potential at lateral position y_ — sum of left and right boundary fields."""
        yr_ = c2*xd**2 + APF_C0R
        yl_ = c2*xd**2 + APF_C0L

        def _U(yb, by_, s):
            # Distance from y_ to the boundary curve, measured along the perpendicular
            inner = np.sqrt(((y_-by_)/m - xd)**2 + (yb-y_)**2 + 1e-9)
            # Morse potential: 0 at centre, rises to A near boundary
            return APF_A*(1 - np.exp(-np.clip(s*APF_B*inner, -50, 50)))**2

        sr = np.sign(y_-yr_) if abs(y_-yr_) > 1e-6 else  1.0
        sl = np.sign(y_-yl_) if abs(y_-yl_) > 1e-6 else -1.0
        return _U(yr_, by_r, sr) + _U(yl_, by_l, sl)

    # Numerical gradient: (U(y+ε) - U(y-ε)) / 2ε
    # Negative because the force pushes AWAY from high potential (toward centre)
    return 0.0, float(-(Utotal(y_veh+eps) - Utotal(y_veh-eps)) / (2*eps))


def compute_obstacle_force(good_new, flow_vecs, ttc_arr, grad_x, grad_y):
    """
    Computes the repulsive force from all detected obstacles.
    Direction: gradient of the Gaussian-smoothed obstacle map (points away from obstacles)
    Magnitude: scaled by gamma, divided by sum of TTCs (closer obstacles push harder)
    """
    if good_new is None or len(good_new) == 0:
        return 0.0, 0.0

    pts = good_new.reshape(-1, 2).astype(int)
    xs  = np.clip(pts[:,0], 0, WIDTH-1)
    ys  = np.clip(pts[:,1], 0, HEIGHT-1)

    # Use TTC as weights — smaller TTC means more urgent, larger contribution
    tv  = ttc_arr if len(ttc_arr)==len(xs) else np.ones(len(xs))
    ts  = max(float(tv.sum()), 1e-6)  # total TTC sum (denominator)
    n   = max(len(xs), 1)             # number of points

    # Sum up gradients at all tracked points, normalise by count and TTC sum
    return (float(GAMMA_REP/n * grad_x[ys,xs].sum() / ts),
            float(GAMMA_REP/n * grad_y[ys,xs].sum() / ts))


def compute_ttc(good_new, flow_vecs, xFOE, yFOE):
    """
    Computes Time-to-Contact for each tracked point.
    TTC = (distance from FOE) / (flow magnitude)
    Small TTC = imminent contact. Large TTC = far away in time.
    """
    if good_new is None or len(good_new) == 0:
        return np.array([])

    pts = good_new.reshape(-1, 2)
    vxy = flow_vecs.reshape(-1, 2)

    # Numerator: pixel distance from FOE
    # Denominator: flow speed (+ small epsilon to avoid division by zero)
    return np.sqrt((pts[:,0]-xFOE)**2 + (pts[:,1]-yFOE)**2) / \
           (np.sqrt(vxy[:,0]**2 + vxy[:,1]**2) + 1e-6)


def compute_total_force(car_id, xFOE, good_new, flow_vecs, ttc_arr,
                        grad_x, grad_y, yaw, centre_FY, dodge_FY):
    """
    Combines all forces into a single resultant force vector.
    Then rotates it from the vehicle's local frame into the global world frame.

    Forces:
      Fatt  : pulls car toward goal (forward + centred)
      Fo    : pushes car away from obstacles (from optical flow)
      Fr    : pushes car away from road boundaries (Morse potential)
      centre: keeps car in lane centre (PD controller)
      dodge : steers around detected obstacles (raycast-based)
    """
    pos, _  = p.getBasePositionAndOrientation(car_id)

    # Attractive force — always push forward (x) and toward y=0 (centre)
    Fatt_x = ALPHA_ATT * 1.0
    Fatt_y = ALPHA_ATT * (-float(pos[1]))

    # Obstacle repulsive force (from optical flow gradient)
    Fo_x, Fo_y = compute_obstacle_force(good_new, flow_vecs, ttc_arr, grad_x, grad_y)

    # Road boundary repulsive force (from Morse potential)
    Fr_x, Fr_y = compute_road_potential_field(car_id, xFOE)

    # Sum all forces in vehicle local frame (equations 19, 20 from paper)
    FXT = Fatt_x - Fo_x - LAMBDA_X * Fr_x
    FYT = Fatt_y - Fo_y - LAMBDA_Y * Fr_y + centre_FY + dodge_FY

    # Rotate from vehicle frame to global world frame (equation 21 from paper)
    # This accounts for the car's current heading direction
    c = np.cos(yaw);  s = np.sin(yaw)
    return float(c*FXT + s*FYT), float(-s*FXT + c*FYT)


# ═════════════════════════════════════════════════════════════════════════════
# GTSMC CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════

def gtsmc_lateral(car_id, FX0, FY0, dt):
    """
    Gradient Tracking Sliding Mode Controller for steering.

    1. Compute desired heading from the resultant force direction
    2. Compute heading error
    3. Compute sliding manifold s = cr*error + error_rate
    4. Apply bang-bang steering rate: always steer at max rate toward manifold
    """
    global delta_f

    # Desired heading angle = direction of the resultant force vector (equation 27)
    # arctan2(FY, FX) gives the angle of the force vector
    # + 1e-9 prevents division by zero when FX is exactly 0
    psi_d = float(np.arctan2(FY0, FX0 + 1e-9))

    # Get the car's actual current heading from its orientation quaternion
    _, orn = p.getBasePositionAndOrientation(car_id)
    psi    = float(p.getEulerFromQuaternion(orn)[2])  # yaw angle

    # Heading error wrapped to [-π, π] so we always take the shortest turn
    psi_e = (psi - psi_d + np.pi) % (2*np.pi) - np.pi

    # Sliding manifold (equation 28): s = cr * error + error_rate
    # Dividing by dt approximates the derivative of the error
    sr = CR * psi_e + psi_e / max(dt, 1e-4)

    # Bang-bang steering rate: steer left at max rate if s > 0, right if s < 0
    # Integrate to get steering angle (equations 29, 30)
    delta_f = float(np.clip(delta_f + (-U0 * float(np.sign(sr))) * dt,
                            -STEER_MAX, STEER_MAX))
    return delta_f


def gtsmc_longitudinal(obs_speed, dt):
    """
    Sliding mode controller for speed.
    Drives current speed toward the target speed (obs_speed).
    Bang-bang: always accelerate or brake at maximum rate.
    """
    global speed_cur

    # Longitudinal sliding manifold (equation 31): s = cl * speed - target_speed
    sl = CL * speed_cur - obs_speed

    # Bang-bang acceleration: max accelerate if too slow, max brake if too fast
    speed_cur = float(np.clip(speed_cur + (-A0 * float(np.sign(sl))) * dt,
                              0.5, SPEED_REF + 1.0))
    return speed_cur


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    global prev_gray, prev_pts, delta_f, speed_cur

    # Initialise the PyBullet simulation (road, obstacles, car)
    car_id, steer_j, motor_j = setup_simulation()
    print("[VPF] Starting")

    last_t = time.time()  # used to compute dt (time step) each frame

    while True:
        p.stepSimulation()  # advance physics by one timestep

        now    = time.time()
        dt     = max(now - last_t, 1e-4)  # time since last frame (seconds)
        last_t = now

        # Get the camera image from the car's viewpoint
        gray, rgb = get_camera_image(car_id)

        # On the very first frame, just store the image and wait for the next one
        if prev_gray is None:
            prev_gray = gray
            prev_pts  = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            continue

        # ── OPTICAL FLOW ──────────────────────────────────────────────────────

        # Track feature points from previous frame to current frame
        good_old, good_new, flow_vecs, prev_pts = compute_optical_flow(prev_gray, gray, prev_pts)
        prev_gray = gray  # store current frame for next iteration

        # Find the Focus of Expansion (direction of travel)
        xFOE, yFOE = compute_foe(good_old, flow_vecs)

        # Compute Time-to-Contact for each tracked point
        ttc_arr = compute_ttc(good_new, flow_vecs, xFOE, yFOE)

        # Build the obstacle map and compute its gradient
        obs_mask, grad_x, grad_y = build_obstacle_map(good_new, flow_vecs, xFOE, yFOE)

        # ── CAR STATE ─────────────────────────────────────────────────────────

        pos_now, orn = p.getBasePositionAndOrientation(car_id)
        car_x  = float(pos_now[0])   # forward position
        car_y  = float(pos_now[1])   # lateral position (0 = road centre)
        yaw    = float(p.getEulerFromQuaternion(orn)[2])  # heading angle

        # ── RAYCASTING ────────────────────────────────────────────────────────

        # Cast 5 rays ahead and get closest obstacle information
        min_dist, hit_world_y, dist_left, dist_right, ray_hits = raycast_obstacle(car_id)

        # Convert distance to threat level
        raw_threat       = classify_raw_threat(min_dist)

        # Apply hysteresis to prevent flickering between threat levels
        confirmed_threat = apply_hysteresis(raw_threat)

        # ── FORCES ────────────────────────────────────────────────────────────

        # Compute dodge force (lateral steering around the obstacle)
        dodge_FY, obs_speed, in_cooldown, centre_boost, bleed_integrator = \
            compute_dodge_force(confirmed_threat, hit_world_y, car_x, car_y, min_dist, dt)

        # Compute lane centring force (keeps car at y=0)
        centre_FY, abs_y = compute_centre_force(car_id, dt, boost=centre_boost)

        # Slowly bleed the steering integrator during re-centring to prevent wind-up
        if bleed_integrator and abs_y > 0.05:
            delta_f *= 0.92  # reduce steering accumulation by 8% each frame

        # Combine all forces into a single resultant in global frame
        FX0, FY0 = compute_total_force(car_id, xFOE, good_new, flow_vecs, ttc_arr,
                                       grad_x, grad_y, yaw, centre_FY, dodge_FY)

        # ── CONTROLLER ────────────────────────────────────────────────────────

        # Compute steering angle from the GTSMC lateral controller
        steering = gtsmc_lateral(car_id, FX0, FY0, dt)

        # Compute target speed from the GTSMC longitudinal controller
        speed    = gtsmc_longitudinal(obs_speed, dt)

        # ── EMERGENCY OVERRIDE ────────────────────────────────────────────────

        # At 3.5m or less, bypass the GTSMC entirely — maximum steering immediately
        if confirmed_threat == "emergency":
            if abs(hit_world_y) < 0.08:
                # Obstacle is centred — pick direction based on which side has more space
                dodge_dir = 1.0 if dist_left >= dist_right else -1.0
            else:
                # Dodge away from whichever side the obstacle is on
                dodge_dir = -1.0 if hit_world_y > 0 else 1.0
            steering = dodge_dir * STEER_MAX
            delta_f  = steering  # override the integrator too
            speed    = max(speed, OBS_SPEED_BRAKE)

        # ── HARD BOUNDARY GUARD ───────────────────────────────────────────────

        # Last resort: if the car reaches the road edge, steer hard back
        boundary_active = abs(car_y) > BOUNDARY_Y
        if boundary_active:
            steering = -float(np.sign(car_y)) * BOUNDARY_STEER  # push back toward centre
            delta_f  = steering
            speed    = min(speed, BOUNDARY_SPEED)  # slow down too

        # ── ACTUATION ─────────────────────────────────────────────────────────

        # Apply steering angle to all steering joints
        for j in steer_j:
            p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL,
                                    targetPosition=steering, force=250, positionGain=0.3)

        # Apply target wheel speed to all motor joints
        for j in motor_j:
            p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL,
                                    targetVelocity=speed, force=1200)

        # ── VISUALISATION ─────────────────────────────────────────────────────

        # Start with the camera image in BGR colour
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Draw green arrows showing optical flow vectors
        if good_new is not None and good_old is not None:
            for n_pt, o_pt in zip(good_new.reshape(-1,2), good_old.reshape(-1,2)):
                cv2.arrowedLine(vis, (int(o_pt[0]),int(o_pt[1])),
                                (int(n_pt[0]),int(n_pt[1])), (0,200,0), 1, tipLength=0.4)

        # Draw a red dot at the Focus of Expansion (direction of travel)
        cv2.circle(vis, (int(xFOE), int(yFOE)), 6, (0,0,255), -1)

        # Colour code for each threat level (green=clear, yellow=warn, orange=brake, red=emergency)
        threat_col = {"clear":(0,255,0), "warn":(0,255,255),
                      "brake":(0,165,255), "emergency":(0,0,255)}
        col = threat_col.get(confirmed_threat, (0,255,0))

        # Draw raycast hit bars at the bottom of the image
        for (hit_dist, ang_deg, did_hit) in ray_hits:
            bar_x = int(WIDTH/2 + ang_deg * (WIDTH / (2*max(RAY_ANGLES_DEG))))
            bar_h = int((1.0 - hit_dist / RAY_LENGTH) * 60)  # taller bar = closer obstacle
            cv2.rectangle(vis, (bar_x-6, HEIGHT-10-bar_h), (bar_x+6, HEIGHT-10),
                          (0,0,255) if did_hit else (0,200,0), -1)

        # Draw vertical centre line to help judge lateral position
        cv2.line(vis, (WIDTH//2, 0), (WIDTH//2, HEIGHT), (255,255,255), 1)

        # Overlay the obstacle mask in red (semi-transparent)
        mo = np.zeros_like(vis);  mo[:,:,2] = obs_mask
        vis = cv2.addWeighted(vis, 1.0, mo, 0.3, 0)

        # Determine current mode label for the HUD
        status = "BOUNDARY!" if boundary_active else \
                 ("DODGE"    if _dodge_active else \
                 ("RECENTRE" if in_cooldown else "CENTRE"))

        # Build the heads-up display text
        hud = [
            f"Mode:    {status}",
            f"Threat:  {confirmed_threat.upper()}  (raw:{raw_threat})",
            f"Dist:    {min_dist:.1f}m  hit_y:{hit_world_y:+.2f}  L:{dist_left:.1f}  R:{dist_right:.1f}",
            f"Speed:   {speed:.1f} m/s",
            f"Steer:   {steering:+.3f} rad",
            f"Car Y:   {car_y:+.3f} m   target:{_dodge_target_y:+.2f}",
        ]

        # Draw each HUD line with appropriate colour
        for i, txt in enumerate(hud):
            c = col if i == 1 else (0,0,220) if boundary_active and i==0 else (0,255,180)
            cv2.putText(vis, txt, (8, 22+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, c, 1)

        cv2.imshow("VPF Navigation", vis)

        # Exit if Escape or 'q' is pressed
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    # ── CLEANUP ───────────────────────────────────────────────────────────────

    cv2.destroyAllWindows()
    p.disconnect()
    print("[VPF] Done.")


if __name__ == "__main__":
    main()