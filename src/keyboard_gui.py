import threading
import time
import tkinter as tk
import traceback
import cv2
import mediapipe as mp
import numpy as np

# Initialize Tkinter GUI
try:
    root = tk.Tk()
    root.title("Gaze Keyboard")
    root.geometry("800x750")
except Exception as e:
    print(f"Tkinter init error: {e}")
    exit()

# Text display label
text_var = tk.StringVar()
text_label = tk.Label(
    root,
    textvariable=text_var,
    font=("Arial", 24, "bold"),
    bg="lightgray",
    width=40,
    height=2,
    borderwidth=2,
    relief="sunken"
)
text_label.pack(pady=20, padx=10, fill="x")

# Word entry
word_var = tk.StringVar()
word_entry = tk.Entry(
    root,
    textvariable=word_var,
    font=("Arial", 16),
    width=20,
    state='readonly',
    justify="center"
)
word_entry.pack(pady=5)

# Gaze status label
gaze_var = tk.StringVar(value="Gazing at: None")
tk.Label(root, textvariable=gaze_var, font=("Arial", 14)).pack(pady=5)

# Gaze coordinates
coord_var = tk.StringVar(value="Gaze coords: (0, 0)")
tk.Label(root, textvariable=coord_var, font=("Arial", 12)).pack(pady=5)

# Suggestion status
suggestion_status_var = tk.StringVar(value="Suggestions: None")
tk.Label(root, textvariable=suggestion_status_var, font=("Arial", 12)).pack(pady=5)

# Suggestion buttons
suggestion_frame = tk.Frame(root)
suggestion_frame.pack(pady=5)
suggestion_vars = [tk.StringVar(value="") for _ in range(3)]
suggestion_buttons = []
for i in range(3):
    btn = tk.Button(
        suggestion_frame,
        textvariable=suggestion_vars[i],
        font=("Arial", 12),
        width=15,
        command=lambda idx=i: select_suggestion(idx)
    )
    btn.grid(row=0, column=i, padx=5)
    suggestion_buttons.append(btn)

# Button frame for keyboard
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Canvas for gaze visualization
canvas = tk.Canvas(button_frame, width=600, height=400, bg="white")
canvas.grid(row=0, column=0, columnspan=6, rowspan=6, sticky="nsew")
gaze_dot = None

# Create keyboard buttons
buttons = {}
button_regions = {}
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
for i, letter in enumerate(letters):
    row, col = i // 6, i % 6
    btn = tk.Button(
        button_frame,
        text=letter,
        font=("Arial", 16),
        width=5,
        height=2,
        command=lambda l=letter: select_letter(l)
    )
    btn.grid(row=row, column=col, padx=5, pady=5)
    buttons[letter] = btn

# Delete button
del_btn = tk.Button(
    button_frame,
    text="DEL",
    font=("Arial", 16),
    width=5,
    height=2,
    command=lambda: select_letter("DEL")
)
del_btn.grid(row=5, column=5, padx=5, pady=5)
buttons["DEL"] = del_btn

# Curated dictionary for daily needs
curated_words = {
    'wa': ['water', 'want', 'wait'],
    'hel': ['hello', 'help', 'held'],
    'by': ['bye', 'by', 'buy'],
    'to': ['toilet', 'to', 'today'],
    'fo': ['food', 'for', 'follow'],
    'ye': ['yes', 'yet', 'yell'],
    'no': ['no', 'now', 'note'],
    'pl': ['please', 'play', 'plan'],
    'th': ['thank', 'the', 'this']
}

# Global variables
stop_webcam = False
log_file = open("experiment_log.txt", "a")

def update_button_regions():
    """Store button coordinates relative to canvas."""
    try:
        canvas_coords = canvas.winfo_rootx(), canvas.winfo_rooty()
        for letter, btn in buttons.items():
            x = btn.winfo_rootx() - canvas_coords[0]
            y = btn.winfo_rooty() - canvas_coords[1]
            w, h = btn.winfo_width(), btn.winfo_height()
            button_regions[letter] = (x, y, w, h)
        for i, btn in enumerate(suggestion_buttons):
            x = btn.winfo_rootx() - canvas_coords[0]
            y = btn.winfo_rooty() - canvas_coords[1]
            w, h = btn.winfo_width(), btn.winfo_height()
            button_regions[f"SUG{i}"] = (x-20, y-20, w+40, h+40)
        root.after(100, update_button_regions)
    except Exception as e:
        print(f"Update regions error: {e}")
        log_file.write(f"{time.time()}: Update regions error: {e}\n")

def select_letter(letter):
    """Handle letter or delete selection."""
    try:
        current_text = text_var.get()
        current_word = word_var.get()
        if letter == "DEL":
            text_var.set(current_text[:-1])
            word_var.set(current_word[:-1] if current_word else "")
        else:
            text_var.set(current_text + letter)
            word_var.set(current_word + letter)
        buttons[letter].config(bg="yellow")
        root.after(300, lambda: buttons[letter].config(bg="SystemButtonFace"))
        print(f"Selected: {letter}, Text: {text_var.get()}, Word: {word_var.get()}")
        log_file.write(f"{time.time()}: Selected: {letter}, Text: {text_var.get()}, Word: {word_var.get()}\n")
        update_suggestions()
    except Exception as e:
        print(f"Select error: {e}")
        log_file.write(f"{time.time()}: Select error: {e}\n")

def select_suggestion(idx):
    """Handle suggestion selection."""
    try:
        suggestion = suggestion_vars[idx].get()
        if suggestion:
            current_text = text_var.get()
            last_space = current_text.rfind(" ")
            new_text = suggestion if last_space == -1 else current_text[:last_space + 1] + suggestion
            text_var.set(new_text + " ")
            word_var.set("")
            print(f"Selected suggestion: {suggestion}, Text: {text_var.get()}")
            log_file.write(f"{time.time()}: Selected suggestion: {suggestion}, Text: {text_var.get()}\n")
            suggestion_buttons[idx].config(bg="yellow")
            root.after(300, lambda: suggestion_buttons[idx].config(bg="SystemButtonFace"))
            update_suggestions()
    except Exception as e:
        print(f"Suggestion select error: {e}")
        log_file.write(f"{time.time()}: Suggestion select error: {e}\n")

def predict_next_word(current_word):
    """Predict next words."""
    try:
        if not current_word:
            return []
        current_word = current_word.lower()
        if current_word in curated_words:
            suggestions = curated_words[current_word][:3]
            print(f"Suggestions for '{current_word}': {suggestions}")
            log_file.write(f"{time.time()}: Suggestions for '{current_word}': {suggestions}\n")
            return suggestions
        return []
    except Exception as e:
        print(f"Predict error: {e}")
        return []

def update_suggestions():
    """Update suggestion buttons."""
    try:
        current_word = word_var.get()
        suggestions = predict_next_word(current_word)
        suggestion_status_var.set(f"Suggestions: {', '.join(suggestions) if suggestions else 'None'}")
        for i, var in enumerate(suggestion_vars):
            var.set(suggestions[i] if i < len(suggestions) else "")
    except Exception as e:
        print(f"Update suggestions error: {e}")
        log_file.write(f"{time.time()}: Update suggestions error: {e}\n")

def webcam_loop():
    """Process webcam frames for gaze tracking and update GUI."""
    global gaze_dot
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not found.")
            log_file.write(f"{time.time()}: Error: Webcam not found\n")
            root.after(0, root.destroy)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_letter = None
        dwell_time = 0
        dwell_threshold = 1.0

        print("Starting webcam loop")
        log_file.write(f"{time.time()}: Starting webcam loop\n")
        while cap.isOpened() and not stop_webcam:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                log_file.write(f"{time.time()}: Error: Failed to capture frame\n")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            print("Processed frame")
            log_file.write(f"{time.time()}: Processed frame\n")

            current_letter = None
            iris_x, iris_y = 0.0, 0.0
            screen_x, screen_y = 0, 0

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_iris = face_landmarks.landmark[468]
                right_iris = face_landmarks.landmark[473]

                iris_x = (left_iris.x + right_iris.x) / 2
                iris_y = (left_iris.y + right_iris.y) / 2
                print(f"Iris: ({iris_x:.3f}, {iris_y:.3f})")
                log_file.write(f"{time.time()}: Iris: ({iris_x:.3f}, {iris_y:.3f})\n")

                screen_x = int((1.0 - iris_x) * canvas.winfo_width() * 1.5)
                screen_y = int(iris_y * canvas.winfo_height() * 1.5)
                screen_x = np.clip(screen_x, 0, canvas.winfo_width() - 1)
                screen_y = np.clip(screen_y, 0, canvas.winfo_height() - 1)
                print(f"Screen: ({screen_x}, {screen_y})")
                log_file.write(f"{time.time()}: Screen: ({screen_x}, {screen_y})\n")

                if gaze_dot:
                    canvas.delete(gaze_dot)
                gaze_dot = canvas.create_oval(
                    screen_x - 5, screen_y - 5,
                    screen_x + 5, screen_y + 5,
                    fill="green"
                )
                print(f"Gaze dot created at ({screen_x}, {screen_y})")
                log_file.write(f"{time.time()}: Gaze dot created at ({screen_x}, {screen_y})\n")

                update_button_regions()
                for letter, (bx, by, bw, bh) in button_regions.items():
                    if bx <= screen_x <= bx + bw and by <= screen_y <= by + bh:
                        current_letter = letter
                        if letter == prev_letter:
                            dwell_time += 1 / 30
                            if dwell_time >= dwell_threshold:
                                root.after(0, lambda l=letter: select_letter(l))
                                dwell_time = 0
                        else:
                            dwell_time = 0
                            prev_letter = letter
                        root.after(0, lambda: buttons[letter].config(bg="lightblue"))
                    else:
                        root.after(0, lambda l=letter: buttons[l].config(bg="SystemButtonFace"))

            gaze_var.set(f"Gazing at: {current_letter if current_letter else 'None'}")
            coord_var.set(f"Gaze coords: ({iris_x:.2f}, {iris_y:.2f})")
            root.update()

            cv2.imshow("Webcam - Gaze Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Exiting webcam loop")
        log_file.write(f"{time.time()}: Exiting webcam loop\n")
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        root.after(0, root.destroy)
    except Exception as e:
        print(f"Webcam loop error: {e}")
        log_file.write(f"{time.time()}: Webcam loop error: {e}\n")
        traceback.print_exc()
        root.after(0, root.destroy)

def cleanup():
    """Clean up resources."""
    global stop_webcam
    try:
        stop_webcam = True
        root.destroy()
        log_file.close()
    except Exception as e:
        print(f"Cleanup error: {e}")

try:
    print("Starting webcam thread")
    log_file.write(f"{time.time()}: Starting webcam thread\n")
    webcam_thread = threading.Thread(target=webcam_loop, daemon=True)
    webcam_thread.start()
    print("Starting GUI")
    log_file.write(f"{time.time()}: Starting GUI\n")
    root.after(100, update_button_regions)
    root.protocol("WM_DELETE_WINDOW", cleanup)
    root.mainloop()
except Exception as e:
    print(f"Startup error: {e}")
    log_file.write(f"{time.time()}: Startup error: {e}\n")
    cleanup()