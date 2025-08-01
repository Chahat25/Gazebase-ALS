import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not found.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting webcam. Look around (left, right, up, down). Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw all landmarks
            mp_drawing.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )
            # Highlight iris and eye corners
            h, w, _ = frame_bgr.shape
            for idx in [468, 473, 33, 133]:  # Left iris, right iris, left eye corners
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_bgr, (x, y), 3, (0, 0, 255), -1)
                cv2.putText(frame_bgr, str(idx), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Log iris landmarks
            left_iris = face_landmarks.landmark[468]
            right_iris = face_landmarks.landmark[473]
            iris_x = (left_iris.x + right_iris.x) / 2
            iris_y = (left_iris.y + right_iris.y) / 2
            print(f"Iris: (x={iris_x:.3f}, y={iris_y:.3f}), Left iris: ({left_iris.x:.3f}, {left_iris.y:.3f}), Right iris: ({right_iris.x:.3f}, {right_iris.y:.3f})")
    else:
        print("No face detected")

    # Show frame
    cv2.imshow("Landmark Test", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("Webcam closed.")