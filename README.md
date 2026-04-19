# 🔐 Face Authentication System with Liveness Detection

## 📌 Project Overview

This project is a **Real-Time Face Authentication System** that provides secure biometric authentication using **face recognition, liveness detection, and embedding-based verification**.

The system ensures that only **real human users** can authenticate by preventing spoofing attacks such as photos or videos.

---

## 🎯 Objective

* Enable secure face-based authentication
* Prevent spoofing using liveness detection
* Ensure privacy by storing only embeddings
* Build a scalable real-time authentication system

---

## 🚀 Key Features

### 🔹 Real-Time Image Capture

* Captures live image using camera
* Ensures real-time interaction
* Prevents static image attacks

---

### 🔹 Background Removal & Preprocessing

* Removes background
* Applies **white background normalization**
* Improves face clarity and consistency

---

### 🔹 Face Detection

* Uses **InsightFace ONNX models**
* Detects valid face regions
* Stops process if no face is detected

---

### 🔹 Liveness Detection (Security Core 🔥)

* Detects real human presence
* Uses:

  * Facial motion
  * Frame consistency
* Randomized liveness checks prevent replay attacks

---

### 🔹 Face Recognition (Embedding-Based)

* Uses **ArcFace model**
* Converts face into embeddings (vector form)
* Matches embeddings instead of storing images

---

### 🔹 API-Based Authentication

* Built using **FastAPI**

**Endpoints:**

* `/enroll` → Register user
* `/authenticate` → Verify user

---

### 🔹 Vector Database (FAISS)

* Stores facial embeddings
* Fast similarity search
* No raw images stored

---

### 🔹 Smart Decision Logic

Authentication is granted only if:

* Face detected ✔
* Liveness passed ✔
* Similarity ≥ 0.60 ✔
* Confidence ≥ 0.70 ✔

---

### 🔹 Session Confidence Formula

```id="conf_formula"
Confidence = (0.8 × Similarity Score) + (0.2 × Image Quality)
```

---

### 🔹 Progressive Lockout System

* Tracks failed attempts
* Temporarily blocks user after multiple failures
* Prevents brute-force attacks

---

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV
* **Face Recognition:** InsightFace (ArcFace)
* **Runtime:** ONNX Runtime
* **Database:** FAISS (Vector DB)
* **Backend:** FastAPI
* **Processing:** NumPy

---

## 📂 Project Structure

```id="proj_struct"
Face-Authentication-System/
│── main.py                # Main pipeline
│── face_align.py          # Face alignment
│── preprocessing.py       # Image processing
│── liveness.py            # Liveness detection
│── recognition.py         # Embedding generation
│── api.py                 # FastAPI endpoints
│── requirements.txt       # Dependencies
│── output/                # Generated images
```

---

## ▶️ How to Run Locally

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Run Main System

```bash
python main.py
```

---

### 3️⃣ Run API (Optional)

```bash
python -m uvicorn api:app --reload
```

---

## 🧪 Testing & Output

✔ Face alignment completed
✔ Background removal applied
✔ Outputs saved successfully

Example output:

* Face aligned image
* Structured image
* Processed images stored with timestamp

---

## 📸 Output Images
**Fig 1:** System Execution and Output Generation

<img width="944" height="170" alt="Screenshot 2026-04-19 192028" src="https://github.com/user-attachments/assets/0facd6eb-27cc-44a7-8423-2d95814b5d6f" />

**Fig 2:** Face Alignment and Structured Output

<img width="942" height="151" alt="Screenshot 2026-04-19 191846" src="https://github.com/user-attachments/assets/25f2a795-0c87-4ea2-b4a5-45e9ced6817d" />

**Fig 3:** Image Capture and Background Removal Process (Final Output)

<img width="903" height="767" alt="Screenshot 2026-04-19 192217" src="https://github.com/user-attachments/assets/c87ca673-10f1-4221-b30f-d03ba50c5062" />


---

## 🔄 System Workflow

```id="workflow"
Image Capture → Preprocessing → Face Detection → Liveness Check → Embedding → Matching → Decision
```

---

## 🔐 Security Features

* Liveness detection prevents spoofing
* Embedding-only storage (no raw images)
* Progressive lockout system
* Randomized verification flow

---

## ⚠️ Limitations

* Requires camera access
* Performance depends on hardware
* Lighting conditions may affect detection

---

## 🔐 Privacy & Data Handling

* No raw images stored
* No base64 storage
* Only embeddings saved
* Temporary data deleted automatically

---

## 🌍 Deployment

* Can be deployed using:

  * VPS
  * FastAPI
  * Nginx

---

## 🙋‍♀️ Author

**Khushi Kasliwal**

📧 Email: [khushikasliwal4@gmail.com](mailto:khushikasliwal4@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/khushi-kasliwal-953692260/

---
