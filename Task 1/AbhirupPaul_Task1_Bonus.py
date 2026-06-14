import numpy as np
import cv2
import time
from scipy.interpolate import LinearNDInterpolator


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_flow(img, flow, step=16):
    """
    Draws arrows on the image showing the direction of motion.
    Samples the flow field on a grid every 'step' pixels — not every pixel,
    because drawing 300,000 arrows would be unreadable.
    """
    h, w = img.shape[:2]

    # Create a grid of (x, y) sample points spaced 'step' pixels apart
    # mgrid produces two arrays — one for y positions, one for x positions
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)

    # Look up the flow vector (fx, fy) at each grid point
    fx, fy = flow[y, x].T

    # Build line segments: from (x, y) to (x-fx, y-fy)
    # We subtract because flow points to where the pixel CAME FROM, not where it's going
    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)  # round to integer pixel positions

    # Convert grayscale to colour so we can draw green arrows
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw each arrow line in green
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    # Draw a small green dot at the start of each arrow (the current position)
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):
    """
    Visualises the full dense flow field using colour:
      - HUE   (colour)     = direction of motion (which way the pixel is moving)
      - VALUE (brightness) = speed of motion (how fast the pixel is moving)
    Stationary pixels appear black. Fast-moving pixels appear bright.
    This gives a whole-image view of the motion field at a glance.
    """
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    # Compute angle of each flow vector — this becomes the colour (hue)
    # arctan2 gives angle in radians, + pi shifts range from [-pi,pi] to [0, 2pi]
    ang = np.arctan2(fy, fx) + np.pi

    # Compute magnitude (speed) of each flow vector — this becomes brightness
    v = np.sqrt(fx * fx + fy * fy)

    # Build a blank HSV image (hue, saturation, value)
    hsv = np.zeros((h, w, 3), np.uint8)

    # Set hue: OpenCV hue range is 0-180, so scale angle accordingly
    hsv[..., 0] = ang * (180 / np.pi / 2)

    # Set saturation to maximum (255) — full colour everywhere
    hsv[..., 1] = 255

    # Set brightness from speed — multiply by 4 to make slow motion visible,
    # cap at 255 so we don't overflow the uint8 range
    hsv[..., 2] = np.minimum(v * 4, 255)

    # Convert HSV back to BGR so OpenCV can display it
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# SPARSE TO DENSE INTERPOLATION
# ─────────────────────────────────────────────────────────────────────────────

def sparse_to_dense(good_old, flow_vectors, h, w):
    """
    Takes a small set of tracked points and their flow vectors,
    and fills in a flow value for EVERY pixel in the image.

    Uses Delaunay triangulation — connects all tracked points into triangles,
    then linearly interpolates within each triangle.
    Pixels outside the convex hull of tracked points get flow = 0.

    This is much better than blurring a sparse flow image, because blurring
    would mix real flow values with zero-filled regions and dilute the results.
    """

    # Reshape tracked point positions to (N, 2) — N points, each with x and y
    pts = good_old.reshape(-1, 2)

    # Separate the x and y components of each flow vector
    fx = flow_vectors[:, 0]
    fy = flow_vectors[:, 1]

    # Create a grid of ALL pixel coordinates in the image
    # grid_x[row, col] = col  (x coordinate)
    # grid_y[row, col] = row  (y coordinate)
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the grid into a list of (x, y) query points — one per pixel
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Build a Delaunay triangulation from tracked points and
    # create an interpolator for the x-component of flow
    interp_fx = LinearNDInterpolator(pts, fx, fill_value=0.0)

    # Same for y-component of flow
    interp_fy = LinearNDInterpolator(pts, fy, fill_value=0.0)

    # Query both interpolators at every pixel in the image
    dense_fx = interp_fx(grid_pts).reshape(h, w).astype(np.float32)
    dense_fy = interp_fy(grid_pts).reshape(h, w).astype(np.float32)

    # Stack x and y components into a single (h, w, 2) flow array
    return np.dstack([dense_fx, dense_fy])


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Process at half resolution for speed
SCALE = 0.5

# Lucas-Kanade optical flow settings
# winSize  : search window around each point
# maxLevel : number of pyramid levels for handling different motion speeds
# criteria : stop after 7 iterations OR when improvement < 0.05
lk_params = dict(
    winSize=(13, 13),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.05)
)

# Corner detection settings — intentionally loose to get MANY points
# More points = better coverage for interpolation across the whole image
# maxCorners   : up to 500 points (vs 20 in sparse — we want dense coverage)
# qualityLevel : very low (0.01) — accept almost any corner, not just strong ones
# minDistance  : corners can be as close as 5 pixels apart
feature_params = dict(
    maxCorners=500,
    qualityLevel=0.01,
    minDistance=5,
    blockSize=5
)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SETUP
# ─────────────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture("OPTICAL_FLOW.mp4")

# Read the first frame to initialise everything
suc, prev = cap.read()
if not suc:
    print("Error: Could not open video")
    exit()

# Shrink to half size
prev     = cv2.resize(prev, None, fx=SCALE, fy=SCALE)

# Convert to grayscale for optical flow
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Detect initial corners to start tracking
p0 = cv2.goodFeaturesToTrack(prevgray, **feature_params)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — runs once per video frame
# ─────────────────────────────────────────────────────────────────────────────

while True:
    # Read the next frame
    suc, img = cap.read()
    if not suc:
        break  # end of video

    # Shrink to half size
    img  = cv2.resize(img, None, fx=SCALE, fy=SCALE)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape  # image dimensions needed for interpolation

    start = time.time()  # start timer for FPS calculation

    # Start with a blank (all-zero) dense flow field
    flow_dense = np.zeros((h, w, 2), dtype=np.float32)

    # Only proceed if we have at least 10 tracked points
    if p0 is not None and len(p0) >= 10:

        # ── PYRAMIDAL LK FORWARD TRACKING ─────────────────────────────────────

        # Track each corner from the previous frame (prevgray) to the current (gray)
        # p1  : new predicted positions for each corner
        # st  : status — 1 if tracking succeeded, 0 if the point was lost
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)

        # Keep only points that were successfully tracked
        good_new = p1[st.ravel() == 1]
        good_old = p0[st.ravel() == 1]

        # Need at least 4 points to build a valid triangulation
        if len(good_old) >= 4:
            # Compute the actual flow vector for each tracked point:
            # flow = new_position - old_position
            flow_vectors = (good_new - good_old).reshape(-1, 2)

            # Interpolate sparse flow vectors to cover every pixel in the image
            flow_dense = sparse_to_dense(good_old, flow_vectors, h, w)

        # ── FEATURE REFRESH ───────────────────────────────────────────────────

        # If too many points were lost, re-detect from scratch
        if len(good_new) < 50:
            p0 = cv2.goodFeaturesToTrack(gray, **feature_params)
        else:
            # Otherwise, continue tracking the successfully tracked points
            p0 = good_new.reshape(-1, 1, 2)

    else:
        # Not enough points — detect fresh corners to start tracking
        p0 = cv2.goodFeaturesToTrack(gray, **feature_params)

    # Store current frame as previous for the next iteration
    prevgray = gray

    # ── TIMING ────────────────────────────────────────────────────────────────

    end = time.time()
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")

    # ── VISUALISATION ─────────────────────────────────────────────────────────

    # Draw motion arrows on the grayscale frame
    flow_vis = draw_flow(gray, flow_dense)

    # Draw the full HSV colour motion map
    hsv_vis  = draw_hsv(flow_dense)

    # Overlay the FPS counter on the arrow view
    cv2.putText(flow_vis, f"{fps:.1f} FPS", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show both visualisation windows
    cv2.imshow("Dense Flow (Pyramidal LK)", flow_vis)
    cv2.imshow("Dense Flow HSV", hsv_vis)

    # Wait 5ms — quit if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

cap.release()            # release the video file
cv2.destroyAllWindows()  # close all display windows