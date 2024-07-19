import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog

# Initialize the main window
root = tk.Tk()
root.title("Face Recognition")

# Create a folder for models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Function to collect face models
def collect_models():
    username = simpledialog.askstring("Input", "Please enter your username:")
    if not username:
        messagebox.showwarning("Input Error", "Please enter a username.")
        return

    user_folder = os.path.join('models', username)
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    count = 0
    collected_images = []
    prev_frame = None

    while count < 200:  # Collect 200 images
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        # Motion Detection: Check if there is motion in the frame
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < 10000:  # Threshold for motion; adjust as needed
                cv2.putText(frame, "Static Image Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow('Collecting Models', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

        prev_frame = gray

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            collected_images.append(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'Collecting {count}/200', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Collecting Models', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save collected images
    for i, img in enumerate(collected_images):
        cv2.imwrite(os.path.join(user_folder, f'{username}_{i+1}.jpg'), img)

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Collection Complete", f"Collected 200 face models for {username}.")

# Function to verify face
def verify_face():
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Training recognizer with existing models
    images, labels = [], []
    label_map = {}
    current_label = 0

    for user_folder in os.listdir('models'):
        user_path = os.path.join('models', user_folder)
        if not os.path.isdir(user_path):  # Skip non-directory files
            continue
        for image_name in os.listdir(user_path):
            image_path = os.path.join(user_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            images.append(image)
            labels.append(current_label)
        label_map[current_label] = user_folder
        current_label += 1

    recognizer.train(images, np.array(labels))

    prev_frame = None
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.3, 5)

        # Motion Detection: Ensure the frame has motion
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            non_zero_count = np.count_nonzero(diff)
            if non_zero_count < 10000:  # Threshold for motion; adjust as needed
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
            if confidence < 70:  # Adjust confidence threshold as needed
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
