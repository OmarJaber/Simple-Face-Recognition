# Simple-Face-Recognition

## Overview

This project provides a simple face recognition application using OpenCV and Python. It includes functionalities for collecting face models and verifying faces in real-time.

## Requirements

Before running the application, you need to install the required Python packages. Use the following command to install them:

1. **Clone the repository**
```bash
git clone https://github.com/OmarJaber/Simple-Face-Recognition.git
```
2. **Navigate to the project directory**
```bash
cd Simple-Face-Recognition
```
3. **Install the required packages**
```bash
pip install -r requirements.txt
```
4. **Run the application**
```bash
python face_recognition.py
```

## How to Use

### **Collect Models**

1. **Click the Collect Models Button**
   - Locate and click the **Collect Models** button on the main interface.

2. **Enter Your Name**
   - An input dialog will appear asking for your name. Enter your name and click **OK**.

3. **Follow Instructions**
   - An Instructions window will pop up with the following guidance:
     - **Move your head at different angles** for better accuracy.

4. **Completion**
   - The application will collect 200 face images. Once this process is completed, a message box will confirm the successful collection.

### **Verify Face**

1. **Click the Verify Face Button**
   - Locate and click the **Verify Face** button on the main interface.

2. **Face Verification**
   - The application will start the camera and display a verification window.
   - It will detect your face and display the recognized username if it matches any collected model.

3. **Close the Verification Window**
   - Close the verification window when you are finished.

Enjoy using the **Simple Face Recognition** application! 😊

