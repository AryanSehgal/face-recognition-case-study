# Face Recognition using OpenCV and Custom KNN

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-latest-blue.svg)

A real-time facial recognition pipeline built with Python, OpenCV, and a custom implementation of the K-Nearest Neighbors (KNN) algorithm from scratch.

This project consists of two main components:
1. **Data Collection Pipeline:** Captures webcam video streams, detects and crops faces, and saves them as flattened NumPy arrays.
2. **Real-Time Recognition System:** Loads the saved face data, compares live webcam faces against the training data using Euclidean distance, and displays bounding boxes with predicted names.

---

## 🚀 Features

* **Custom Machine Learning:** Implements K-Nearest Neighbors (KNN) from scratch without relying on external ML libraries like `scikit-learn`.
* **Real-time Face Detection:** Utilizes OpenCV's Haar Cascades for fast and efficient face detection.
* **Automated Data Processing:** Automatically crops, resizes (100x100), and flattens image data for immediate use in distance calculations.
* **Dynamic Loading:** Automatically reads and categorizes all `.npy` files present in the data folder, using the filenames as class labels.

---

## 🛠️ Prerequisites

Ensure you have Python installed on your system. You will need the following Python libraries:

```bash
pip install opencv-python numpy
```

### Required Files & Directories
Before running the scripts, you must ensure the following exist in your project directory:
1. A folder named **`data`** in the root directory (where the `.npy` files will be saved). 
2. The **`haarcascade_frontalface_alt.xml`** file in the same directory as your scripts. You can download this from the [official OpenCV GitHub repository](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml).

### 📂 Project Structure
```text
├── data/                               # Directory to store user .npy files (Create this!)
├── haarcascade_frontalface_alt.xml     # OpenCV pre-trained Haar Cascade
├── data_collection.py                  # Script 1: Collects training data
└── face_recognition.py                 # Script 2: Runs live recognition
```

---

## 💻 Usage Guide

### Step 1: Generate Training Data
Run the data collection script to register faces into the system. You need to do this for every person you want the model to recognize.

```bash
python data_collection.py
```
* **Process:** The script will prompt you for a name. It will then open your webcam and capture frames.
* **Action:** Look at the camera. The script automatically detects the largest face in the frame and captures an image every 10 frames.
* **Exit:** Press **`q`** to stop capturing. The data will be saved as `<Name>.npy` inside the `./data/` folder.

*Repeat this step for multiple users to build a robust dataset.*

### Step 2: Run the Recognizer
Once you have generated data for at least one person, run the recognition script.

```bash
python face_recognition.py
```
* **Process:** The script loads all `.npy` arrays from the `./data/` folder, combines them into a training set, and assigns labels based on the filenames.
* **Action:** It opens your webcam, detects faces, extracts the Region of Interest (ROI), flattens it, and passes it to the custom KNN function.
* **Output:** A bounding box is drawn around detected faces alongside the predicted name. 
* **Exit:** Press **`q`** to quit the video stream.

---

## 🧠 How It Works

1. **Face Detection:** Both scripts use `cv2.CascadeClassifier` to scan video frames and identify face coordinates `(x, y, w, h)`.
2. **Data Normalization:** Detected faces are cropped with a 10-pixel padding and resized to an absolute 100x100 pixel grid. This ensures all vectors are the same length (10,000 features per frame).
3. **Custom KNN Algorithm:**
   * Calculates the **Euclidean distance** between the flattened incoming live face array and every face array in the saved dataset.
   * Sorts the distances in ascending order.
   * Slices the top `k=5` shortest distances (the 5 nearest neighbors).
   * Determines the most frequent label among those 5 neighbors and outputs it as the final prediction.

---

## ⚠️ Troubleshooting

* **`cv2.error: empty() in function 'detectMultiScale'`**: This means the script cannot find `haarcascade_frontalface_alt.xml`. Ensure the file is in the exact same directory as your Python scripts.
* **`FileNotFoundError: [Errno 2] No such file or directory: './data/'`**: You need to manually create the `data` folder in your project directory before running the data collection script.
* **Inaccurate Predictions**: KNN is sensitive to lighting and background. Ensure the lighting during testing is similar to the lighting during data collection. Add more varied data for better accuracy.
