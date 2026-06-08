# Feature Extraction & Image Matching System

A computer vision project focused on **feature detection, feature description, and image matching** using classical and modern techniques in OpenCV.

The system processes both **grayscale and color images** to extract robust features and evaluate matching performance.

---
## ✨ Key Features

---

## 🔹 1. Feature Detection (Harris Corner Detector)

- Extracts unique keypoints using **Harris Operator**
- Uses λ (lambda) response for corner strength estimation
- Detects stable interest points in grayscale and color images
- Visualizes detected corners on input images
<img width="1704" height="930" alt="image" src="https://github.com/user-attachments/assets/360a26d4-b3c5-4aba-917f-4628f8319dad" />

---

## 🔹 2. Scale-Invariant Feature Extraction (SIFT)

- Extracts scale- and rotation-invariant keypoints using **SIFT**
- Generates robust feature descriptors for each keypoint
- Works on both grayscale and RGB images
- Provides strong invariance to scale and illumination changes
<img width="1707" height="931" alt="image" src="https://github.com/user-attachments/assets/cf51dfe5-836b-409d-9527-4ee9bdf8f4b4" />

---

## 🔹 3. Feature Matching Engine

Matches extracted features between image pairs using two methods:

---

### 📌 A) Sum of Squared Differences (SSD)

- Matches features based on minimum SSD distance
<img width="1702" height="931" alt="image" src="https://github.com/user-attachments/assets/5f0a6418-5c06-4dc4-9a55-3a60c3f44f75" />

---

### 📌 B) Normalized Cross Correlation (NCC)

- Measures similarity using correlation-based matching
<img width="1703" height="934" alt="image" src="https://github.com/user-attachments/assets/592e0471-5a9b-47fb-be62-3c213422313e" />

---

## 🛠️ Tech Stack

- Python   
- OpenCV  
- NumPy  
- Matplotlib  
- Time module (for performance analysis)

---
## 👥 Contributors
|  [Nada Hesham](https://github.com/Nada-Hesham249)  | [Samar Hatem](https://github.com/samarhatem0405) | [Nada Hassan](https://github.com/Nadahassan147) | [Mostafa Ayman](https://github.com/mostafaayman646) | [Amr](https://github.com/Amr2054) |
|-------------------------------|---------------------------|-----------------------------------|-------------------------------|-------------------------------|
