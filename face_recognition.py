import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

# Initialize the main window
root = tk.Tk()
root.title("Face Recognition")

# Set the size of the window and center it on the screen
window_width = 400
window_height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (window_width / 2)
y = (screen_height / 2) - (window_height / 2)
root.geometry(f"{window_width}x{window_height}+{int(x)}+{int(y)}")

# Keep the window on top of all other windows
root.attributes('-topmost', True)

# Create a folder for models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the instructions image
instructions_path = 'images/instructions.jpg'  # Make sure to place this image in your project directory
instructions_image = Image.open(instructions_path)
instructions_image = instructions_image.resize((500, 500))  # Resize as needed
instructions_photo = ImageTk.PhotoImage(instructions_image)

# Convert PIL Image to OpenCV format
instructions_cv_image = np.array(instructions_image)
instructions_cv_image = cv2.cvtColor(instructions_cv_image, cv2.COLOR_RGB2BGR)

# Function to collect face models
def collect_models():
    root.attributes('-topmost', False)
    username = simpledialog.askstring("Input", "Please enter your name:")
    if not username:
        messagebox.showwarning("Input Error", "Please enter your name.")
        return

    npz_file_path = os.path.join('models', f'{username}_faces.npz')

    cap = cv2.VideoCapture(0)
    count = 0
    collected_faces = []
    prev_frame = None

    while count < 200:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < 10000:
                cv2.putText(frame, "Static Image Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Collecting Models', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        prev_frame = gray

        # Overlay the instructions image on the frame
        cv2.putText(frame, "Please move your head angles for better accuracy", (10, instructions_cv_image.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frame[0:instructions_cv_image.shape[0], 0:instructions_cv_image.shape[1]] = instructions_cv_image

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            collected_faces.append(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Collecting {count}/200', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Collecting Models', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    np.savez_compressed(npz_file_path, faces=np.array(collected_faces))

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Collection Complete", f"Collected 200 face models for {username}.")

# Function to verify face
def verify_face():
    root.attributes('-topmost', False)
    
    # Check if there is any training data available
    if not any(file.endswith('_faces.npz') for file in os.listdir('models')):
        messagebox.showwarning("No Training Data", "No training data found. Please collect face models first.")
        return

    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images, labels = [], []
    label_map = {}
    current_label = 0

    for npz_file in os.listdir('models'):
        if npz_file.endswith('_faces.npz'):
            npz_path = os.path.join('models', npz_file)
            data = np.load(npz_path)
            face_images = data['faces']
            images.extend(face_images)
            labels.extend([current_label] * len(face_images))
            label_map[current_label] = npz_file.replace('_faces.npz', '')
            current_label += 1

    if len(images) == 0:
        messagebox.showwarning("No Training Data", "No training data available for recognition.")
        cap.release()
        return

    recognizer.train(images, np.array(labels))

    prev_frame = None

    # Create a window for verification with a close button
    verify_window = tk.Toplevel(root)
    verify_window.title("Face Verification")
    verify_window.geometry("640x480")

    # Function to handle the close button click
    def close_verify_window():
        cap.release()
        cv2.destroyAllWindows()
        verify_window.destroy()

    # Add a close button to the verification window
    close_button = tk.Button(verify_window, text="Close", command=close_verify_window)
    close_button.pack()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < 10000:
                cv2.putText(frame, "Motionless Image Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Verify Face', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        prev_frame = gray

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            label, confidence = recognizer.predict(face)
            if confidence < 70:
                username = label_map.get(label, "Unknown")
            else:
                username = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, username, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Verify Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Elements
collect_button = tk.Button(root, text="Collect Models", command=collect_models)
collect_button.pack(pady=10)

verify_button = tk.Button(root, text="Verify Face", command=verify_face)
verify_button.pack(pady=10)

# Start the GUI loop
root.mainloop()
