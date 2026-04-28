import cv2
import mediapipe as mp
import pyautogui

# Camera setup
cam = cv2.VideoCapture(0)
cam.set(3, 640)   # width
cam.set(4, 480)   # height

# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
)

# Screen size
screen_w, screen_h = pyautogui.size()

# Initial cursor position
screen_x, screen_y = screen_w // 2, screen_h // 2

# Smoothing factor (lower = smoother)
smooth_factor = 0.2

# Frame skipping for performance
frame_count = 0

# Blink cooldown
click_cooldown = 0

while True:
    frame_count += 1
    if frame_count % 2 != 0:
        continue

    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)

    if output.multi_face_landmarks:
        landmarks = output.multi_face_landmarks[0].landmark

        # ===== 👁️ IRIS CENTER (more stable than single point) =====
        iris_points = [474, 475, 476, 477]
        iris_x = sum([landmarks[i].x for i in iris_points]) / 4
        iris_y = sum([landmarks[i].y for i in iris_points]) / 4

        x = int(iris_x * frame_w)
        y = int(iris_y * frame_h)

        # Draw iris center
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # ===== 🖱️ MOUSE MOVEMENT =====
        new_screen_x = screen_w * iris_x
        new_screen_y = screen_h * iris_y

        # Smooth movement
        screen_x += (new_screen_x - screen_x) * smooth_factor
        screen_y += (new_screen_y - screen_y) * smooth_factor

        pyautogui.moveTo(screen_x, screen_y)

        # ===== 👁️ BLINK DETECTION =====
        left_top = landmarks[159]
        left_bottom = landmarks[145]

        # Draw blink points
        x1, y1 = int(left_top.x * frame_w), int(left_top.y * frame_h)
        x2, y2 = int(left_bottom.x * frame_w), int(left_bottom.y * frame_h)
        cv2.circle(frame, (x1, y1), 3, (0, 255, 255), -1)
        cv2.circle(frame, (x2, y2), 3, (0, 255, 255), -1)

        # Blink logic
        if (left_bottom.y - left_top.y) < 0.01:
            if click_cooldown == 0:
                pyautogui.click()
                click_cooldown = 10  # prevent multiple clicks

    # Reduce cooldown
    if click_cooldown > 0:
        click_cooldown -= 1

    # Show window
    cv2.imshow("Eye Controlled Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cam.release()
cv2.destroyAllWindows()