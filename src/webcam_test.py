import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load CNN model
model = load_model(
    r"C:\Users\Chahat\eye_communicator\models\gaze_cnn.h5",
    custom_objects={'mse': MeanSquaredError()}
)

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

gaze_buffer = []
pred_buffer = []
sequence_length = 30
smooth_n = 10  # Increased smoothing
x_min, x_max = -2.0, 2.0  # Initial range
y_min, y_max = -2.5, 2.5  # Covers 2.448

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Iris and eye corners
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            left_eye_outer = face_landmarks.landmark[33]
            left_eye_inner = face_landmarks.landmark[133]

            # Normalize iris position
            eye_width = abs(left_eye_outer.x - left_eye_inner.x)
            if eye_width > 0:
                gaze_x = ((left_iris.x + right_iris.x) / 2 - left_eye_inner.x) / eye_width
                gaze_y = ((left_iris.y + right_iris.y) / 2 - left_eye_inner.y) / eye_width
            else:
                gaze_x, gaze_y = 0.0, 0.0

            # Scale to training range
            gaze_x *= 3.0  # Increased for alignment
            gaze_y *= 3.0

            gaze_buffer.append([gaze_x, gaze_y])
            if len(gaze_buffer) > sequence_length:
                gaze_buffer.pop(0)

            if len(gaze_buffer) == sequence_length:
                X = np.array([gaze_buffer])
                X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
                y_pred = model.predict(X, verbose=0)[0]
                pred_buffer.append(y_pred)
                if len(pred_buffer) > smooth_n:
                    pred_buffer.pop(0)

                # Smooth predictions
                smooth_pred = np.mean(pred_buffer, axis=0) if pred_buffer else y_pred
                print("Predicted gaze (normalized):", smooth_pred)

                # Update calibration dynamically
                x_min = min(x_min, smooth_pred[0])
                x_max = max(x_max, smooth_pred[0])
                y_min = min(y_min, smooth_pred[1])
                y_max = max(y_max, smooth_pred[1])

                # Map to screen
                screen_w, screen_h = w, h
                screen_x = int(((smooth_pred[0] - x_min) / (x_max - x_min + 1e-8)) * screen_w)
                screen_y = int(((smooth_pred[1] - y_min) / (y_max - y_min + 1e-8)) * screen_h)
                screen_x = np.clip(screen_x, 0, screen_w - 1)
                screen_y = np.clip(screen_y, 0, screen_h - 1)
                cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Gaze: ({smooth_pred[0]:.2f}, {smooth_pred[1]:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Range: x=({x_min:.2f}, {x_max:.2f}), y=({y_min:.2f}, {y_max:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Webcam - Gaze Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()