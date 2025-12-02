import cv2
import numpy as np
import time

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = 'Hand Safety POC (Hold RED Object)'

LOWER_COLOR = np.array([0, 100, 100], dtype=np.uint8)
UPPER_COLOR = np.array([10, 255, 255], dtype=np.uint8)

DANGER_ZONE = (900, 150, 1200, 600)

THRESHOLD_DANGER = 25
THRESHOLD_WARNING = 75


def calculate_distance_to_rect(point, rect):
    """Distance from a point to the outside of a rectangle (0 if inside)."""
    px, py = point
    x1, y1, x2, y2 = rect

    # Horizontal distance
    if px < x1:
        dx = x1 - px
    elif px > x2:
        dx = px - x2
    else:
        dx = 0

    # Vertical distance
    if py < y1:
        dy = y1 - py
    elif py > y2:
        dy = py - y2
    else:
        dy = 0

    return np.sqrt(dx * dx + dy * dy)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    current_state = "SAFE"
    hand_pos = None

    print("--- Hand Tracking Safety POC Initialized ---")
    print(f"Running at target resolution {FRAME_WIDTH}x{FRAME_HEIGHT} in full screen.")
    print("Hold a bright RED object (e.g., marker or sticky note) in your hand.")
    print("Move the object near the drawn rectangle boundary.")
    print("Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirroring the frame
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Converting to HSV and threshold for red
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_pos = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 1500:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hand_pos = (cx, cy)

                    cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
                    cv2.circle(frame, hand_pos, 15, (0, 0, 255), -1)

        # State based on hand position and distance
        if hand_pos is not None:
            distance = calculate_distance_to_rect(hand_pos, DANGER_ZONE)

            if distance <= THRESHOLD_DANGER:
                current_state = "DANGER"
            elif distance <= THRESHOLD_WARNING:
                current_state = "WARNING"
            else:
                current_state = "SAFE"
        else:
            current_state = "SAFE"
            distance = -1

        # Danger zone
        x1, y1, x2, y2 = DANGER_ZONE

        if current_state == "DANGER":
            boundary_color = (0, 0, 255)
        elif current_state == "WARNING":
            boundary_color = (0, 165, 255)
        else:
            boundary_color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), boundary_color, 5)

        # State text
        if current_state == "DANGER":
            state_color = (0, 0, 255)
        elif current_state == "WARNING":
            state_color = (0, 165, 255)
        else:
            state_color = (0, 255, 0)

        cv2.putText(
            frame,
            f"State: {current_state}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            state_color,
            3,
            cv2.LINE_AA,
        )

        if current_state == "DANGER":
            cv2.putText(
                frame,
                "!!! DANGER DANGER !!!",
                (width // 2 - 400, height // 2),
                cv2.FONT_HERSHEY_TRIPLEX,
                2.5,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )

        # FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if frame_count % 30 == 0 and elapsed_time > 0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (width - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
