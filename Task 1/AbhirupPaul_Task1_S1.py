import numpy as np
import cv2
import time

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SETUP
# ─────────────────────────────────────────────────────────────────────────────

# Settings for the Lucas-Kanade optical flow tracker
# winSize     : the search window around each point — larger = handles faster motion
# maxLevel    : number of pyramid levels — 3 means 4 scales (0,1,2,3)
#               coarser scales find large motions, finer scales refine them
# criteria    : when to stop iterating — either after 7 steps OR when
#               improvement is less than 0.05, whichever comes first
lk_params = dict(
    winSize=(11, 11),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 7, 0.05)
)

# Settings for the Shi-Tomasi corner detector (finds good points to track)
# maxCorners   : only keep the 20 strongest corners
# qualityLevel : a corner must score at least 30% of the best corner found
#                — this is strict, so only strong, reliable corners are kept
# minDistance  : corners must be at least 10 pixels apart (spread them out)
# blockSize    : neighbourhood size used when computing corner strength
feature_params = dict(
    maxCorners=20,
    qualityLevel=0.3,
    minDistance=10,
    blockSize=7
)

# How many past positions to remember per tracked point (draws a tail/trail)
trajectory_len = 40

# Re-detect new corners every 5 frames to replace lost tracks
detect_interval = 5

# List of all active trajectories — each one is a list of (x,y) positions
trajectories = []

# Counts which frame we are on (used to trigger re-detection)
frame_idx = 0

# Open the video file
cap = cv2.VideoCapture("OPTICAL_FLOW.mp4")

# Process at half resolution to run faster
SCALE = 0.5

# Will hold the mask image used to avoid re-detecting already-tracked areas
mask = None

# Will hold the previous frame in grayscale (needed to compare with current frame)
prev_gray = None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — runs once per video frame
# ─────────────────────────────────────────────────────────────────────────────

while True:
    start = time.time()  # record start time to compute FPS

    # Read the next frame from the video
    suc, frame = cap.read()
    if not suc:
        break  # end of video, stop the loop

    # Shrink the frame to half size for faster processing
    small_frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)

    # Convert to grayscale — optical flow works on intensity, not colour
    frame_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    # Make a copy of the colour frame to draw visualisations onto
    img = small_frame.copy()

    # Skip the very first frame — we need two frames to compute flow
    if prev_gray is None:
        prev_gray = frame_gray
        continue

    # ── OPTICAL FLOW TRACKING ─────────────────────────────────────────────────

    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray  # previous and current grayscale frames

        # Extract the most recent position of each tracked point
        # shape: (N, 1, 2) — required format for OpenCV LK function
        p0 = np.float32([t[-1] for t in trajectories]).reshape(-1, 1, 2)

        # FORWARD PASS: track each point from the previous frame to the current frame
        # p1   : predicted new positions
        # _st  : status — 1 if point was successfully tracked, 0 if lost
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)

        # BACKWARD PASS: track the predicted points back to the previous frame
        # This is a consistency check — if tracking was accurate, we should
        # get back very close to where we started
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        # Compute how far each point drifted after the round-trip (forward then back)
        # If the drift is more than 1 pixel, the track is considered unreliable
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1  # True for reliable tracks, False for unreliable ones

        new_trajectories = []

        # Loop over each trajectory, its new position, and whether it's reliable
        for trajectory, (x, y), good_flag in zip(trajectories, p1.reshape(-1, 2), good):
            if not good_flag:
                continue  # discard this point — it failed the consistency check

            # Add the new position to this point's history trail
            trajectory.append((x, y))

            # Keep only the most recent positions (drop oldest if too long)
            if len(trajectory) > trajectory_len:
                del trajectory[0]

            new_trajectories.append(trajectory)

            # Draw a small red dot at the current position of this point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

        # Replace old trajectories with the updated (cleaned) list
        trajectories = new_trajectories

        # Draw green lines connecting each point's history of positions (the trail)
        cv2.polylines(img, [np.int32(t) for t in trajectories], False, (0, 255, 0))

        # Show how many points are currently being tracked
        cv2.putText(img, 'track count: %d' % len(trajectories),
                    (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # ── RE-DETECT NEW FEATURES ────────────────────────────────────────────────

    # Every 'detect_interval' frames, find new corners to track
    if frame_idx % detect_interval == 0:

        # Create a blank white mask (255 = allowed, 0 = blocked)
        mask = np.zeros_like(frame_gray)
        mask[:] = 255  # allow detection everywhere by default

        # Block out small circles around already-tracked points
        # This prevents detecting the same corner twice
        for x, y in [np.int32(t[-1]) for t in trajectories]:
            cv2.circle(mask, (x, y), 5, 0, -1)  # fill black circle = blocked region

        # Detect new corners — only in unblocked (white) regions of the mask
        p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

        if p is not None:
            # Add each new corner as a fresh single-point trajectory
            for x, y in np.float32(p).reshape(-1, 2):
                trajectories.append([(x, y)])

    frame_idx += 1        # advance the frame counter
    prev_gray = frame_gray  # store current frame as previous for next iteration

    # ── DISPLAY ───────────────────────────────────────────────────────────────

    end = time.time()
    fps = 1 / (end - start)  # frames per second = 1 / time taken

    # Show FPS in the top-left corner
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the main tracking visualisation
    cv2.imshow("Sparse Optical Flow (Fast)", img)

    # Show the detection mask if it exists
    if mask is not None:
        cv2.imshow("Mask", mask)

    # Wait 10ms for a key press — quit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

cap.release()               # release the video file
cv2.destroyAllWindows()     # close all OpenCV windows