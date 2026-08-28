import cv2
import pygame
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = r"best.pt"

CAMERA_ID = 0

CONF_THRESHOLD = 0.20

# Pupil movement required before changing direction
HORIZONTAL_THRESHOLD = 12
VERTICAL_THRESHOLD = 12

# Smoothing
SMOOTHING_FRAMES = 5

# Ball
BALL_SPEED = 7
BALL_RADIUS = 20

# ============================================================
# 2. LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")

# ============================================================
# 3. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ============================================================
# 4. PYGAME INITIALIZATION
# ============================================================

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

screen = pygame.display.set_mode(
    (SCREEN_WIDTH, SCREEN_HEIGHT)
)

pygame.display.set_caption(
    "YOLO Eye Controlled Ball"
)

clock = pygame.time.Clock()

# Ball initial position
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2

# ============================================================
# 5. SMOOTHING BUFFER
# ============================================================

pupil_history = deque(
    maxlen=SMOOTHING_FRAMES
)

# ============================================================
# 6. FIND BEST TWO PUPILS
# ============================================================

def find_pupil_pair(frame):

    results = model.predict(
        frame,
        imgsz=640,
        conf=CONF_THRESHOLD,
        device=0,
        verbose=False
    )

    detections = []

    if results[0].boxes is None:
        return None

    for box in results[0].boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = xyxy

        confidence = float(
            box.conf[0].cpu().numpy()
        )

        # Center of bounding box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append(
            {
                "x": cx,
                "y": cy,
                "conf": confidence
            }
        )

    # Need at least two detections
    if len(detections) < 2:
        return None

    # --------------------------------------------------------
    # Find the best pair
    # --------------------------------------------------------

    best_pair = None
    best_score = -999

    for i in range(len(detections)):

        for j in range(i + 1, len(detections)):

            p1 = detections[i]
            p2 = detections[j]

            x1 = p1["x"]
            y1 = p1["y"]

            x2 = p2["x"]
            y2 = p2["y"]

            horizontal_distance = abs(x1 - x2)
            vertical_distance = abs(y1 - y2)

            # Pupils should be reasonably separated
            if horizontal_distance < 40:
                continue

            # They should not be extremely far apart
            if horizontal_distance > 450:
                continue

            # Eye pupils should have similar vertical position
            if vertical_distance > 80:
                continue

            # Higher confidence is better
            score = (
                p1["conf"]
                + p2["conf"]
            )

            # Penalize vertical mismatch
            score -= vertical_distance * 0.01

            if score > best_score:

                best_score = score

                best_pair = (
                    p1,
                    p2
                )

    return best_pair


# ============================================================
# 7. CALCULATE AVERAGE PUPIL CENTER
# ============================================================

def calculate_pupil_center(pair):

    left_pupil = pair[0]
    right_pupil = pair[1]

    x = (
        left_pupil["x"]
        + right_pupil["x"]
    ) / 2

    y = (
        left_pupil["y"]
        + right_pupil["y"]
    ) / 2

    return x, y


# ============================================================
# 8. ESTIMATE GAZE DIRECTION
# ============================================================

def estimate_gaze(
    pupil_x,
    pupil_y,
    frame_width,
    frame_height
):

    # Frame center
    center_x = frame_width / 2
    center_y = frame_height / 2

    # Difference from center
    dx = pupil_x - center_x
    dy = pupil_y - center_y

    # --------------------------------------------------------
    # DEAD ZONE
    # --------------------------------------------------------

    if (
        abs(dx) < HORIZONTAL_THRESHOLD
        and
        abs(dy) < VERTICAL_THRESHOLD
    ):
        return "CENTER"

    # --------------------------------------------------------
    # Determine dominant movement
    # --------------------------------------------------------

    if abs(dx) > abs(dy):

        if dx < 0:
            return "LEFT"

        else:
            return "RIGHT"

    else:

        if dy < 0:
            return "UP"

        else:
            return "DOWN"


# ============================================================
# 9. SMOOTH GAZE DIRECTION
# ============================================================

def smooth_direction(new_direction):

    direction_history.append(new_direction)

    # Count occurrences
    counts = {}

    for direction in direction_history:

        if direction not in counts:
            counts[direction] = 0

        counts[direction] += 1

    # Most frequent direction
    return max(
        counts,
        key=counts.get
    )


direction_history = deque(
    maxlen=SMOOTHING_FRAMES
)


# ============================================================
# 10. MOVE BALL
# ============================================================

def move_ball(direction):

    global ball_x
    global ball_y

    if direction == "LEFT":

        ball_x -= BALL_SPEED

    elif direction == "RIGHT":

        ball_x += BALL_SPEED

    elif direction == "UP":

        ball_y -= BALL_SPEED

    elif direction == "DOWN":

        ball_y += BALL_SPEED

    # --------------------------------------------------------
    # Keep ball inside screen
    # --------------------------------------------------------

    ball_x = max(
        BALL_RADIUS,
        min(
            SCREEN_WIDTH - BALL_RADIUS,
            ball_x
        )
    )

    ball_y = max(
        BALL_RADIUS,
        min(
            SCREEN_HEIGHT - BALL_RADIUS,
            ball_y
        )
    )


# ============================================================
# 11. MAIN REAL-TIME LOOP
# ============================================================

running = True

current_direction = "CENTER"

while running:

    # ========================================================
    # PYGAME EVENTS
    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

    # ========================================================
    # WEBCAM FRAME
    # ========================================================

    ret, frame = cap.read()

    if not ret:
        print("Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    frame_height, frame_width = frame.shape[:2]

    # ========================================================
    # YOLO PUPIL DETECTION
    # ========================================================

    pupil_pair = find_pupil_pair(frame)

    if pupil_pair is not None:

        # ====================================================
        # LEFT + RIGHT PUPIL
        # ====================================================

        p1 = pupil_pair[0]
        p2 = pupil_pair[1]

        # Sort pupils from left to right
        if p1["x"] < p2["x"]:

            left_pupil = p1
            right_pupil = p2

        else:

            left_pupil = p2
            right_pupil = p1

        # ====================================================
        # DRAW SMALL GREEN PUPIL DOTS
        # ====================================================

        cv2.circle(
            frame,
            (
                int(left_pupil["x"]),
                int(left_pupil["y"])
            ),
            3,
            (0, 255, 0),
            -1
        )

        cv2.circle(
            frame,
            (
                int(right_pupil["x"]),
                int(right_pupil["y"])
            ),
            3,
            (0, 255, 0),
            -1
        )

        # ====================================================
        # PUPIL CENTER
        # ====================================================

        pupil_x, pupil_y = calculate_pupil_center(
            (left_pupil, right_pupil)
        )

        # Average center dot
        cv2.circle(
            frame,
            (
                int(pupil_x),
                int(pupil_y)
            ),
            4,
            (0, 255, 255),
            -1
        )

        # ====================================================
        # STORE FOR SMOOTHING
        # ====================================================

        pupil_history.append(
            (pupil_x, pupil_y)
        )

        # Calculate smoothed pupil position
        smooth_x = np.mean(
            [p[0] for p in pupil_history]
        )

        smooth_y = np.mean(
            [p[1] for p in pupil_history]
        )

        # ====================================================
        # GAZE ESTIMATION
        # ====================================================

        raw_direction = estimate_gaze(
            smooth_x,
            smooth_y,
            frame_width,
            frame_height
        )

        # ====================================================
        # DIRECTION SMOOTHING
        # ====================================================

        current_direction = smooth_direction(
            raw_direction
        )

    else:

        # No pupil detected
        current_direction = "NO PUPIL"

    # ========================================================
    # MOVE BALL
    # ========================================================

    if current_direction != "NO PUPIL":

        move_ball(
            current_direction
        )

    # ========================================================
    # PYGAME DISPLAY
    # ========================================================

    screen.fill(
        (25, 25, 25)
    )

    # Ball
    pygame.draw.circle(
        screen,
        (255, 0, 0),
        (
            int(ball_x),
            int(ball_y)
        ),
        BALL_RADIUS
    )

    # Direction text
    font = pygame.font.Font(
        None,
        40
    )

    text = font.render(
        "GAZE: " + current_direction,
        True,
        (255, 255, 255)
    )

    screen.blit(
        text,
        (30, 30)
    )

    pygame.display.flip()

    # ========================================================
    # CAMERA DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        "GAZE: " + current_direction,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Pupils: "
        + (
            "2"
            if pupil_pair is not None
            else "0"
        ),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "YOLO11n - Real Time Pupil Detection",
        frame
    )

    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        running = False

    # 60 FPS
    clock.tick(60)


# ============================================================
# 12. CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

pygame.quit()

print("Application closed.")