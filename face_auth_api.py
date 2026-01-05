# =========================
# TURN OFF FUTURE WARNINGS
# =========================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# IMPORTS
# =========================
import os, shutil, pickle
import cv2
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File
from insightface.app import FaceAnalysis
from skimage import filters

# =========================================================
# CONFIGURATION  ✅ ALL UPDATES APPLIED HERE
# =========================================================

# Temporary folder to store images (auto-deleted after use)
UPLOAD_DIR = "runtime_images"

# Vector database files
DB_PATH = "prod_face_db.index"
MAP_PATH = "prod_user_map.pkl"

# Face embedding dimension
VECTOR_DIM = 512

# Authentication decision threshold (stricter security)
THRESHOLD = 0.70

# Face detection resolution (balanced accuracy & speed)
DET_SIZE = (512, 512)

# Session confidence weights
SIMILARITY_WEIGHT = 0.8
QUALITY_WEIGHT = 0.2

# =========================================================
# SETUP
# =========================================================
os.makedirs(UPLOAD_DIR, exist_ok=True)

# =========================
# LOAD ARCFACE MODEL (ONE TIME)
# =========================
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=DET_SIZE)

# =========================
# LOAD / CREATE FAISS DB
# =========================
if os.path.exists(DB_PATH) and os.path.exists(MAP_PATH):
    index = faiss.read_index(DB_PATH)
    user_map = pickle.load(open(MAP_PATH, "rb"))
else:
    index = faiss.IndexFlatIP(VECTOR_DIM)
    user_map = {}

# =========================
# UTILITY FUNCTIONS
# =========================
def get_embedding(img_path):
    img = cv2.imread(img_path)
    faces = face_app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")

    emb = faces[0].embedding
    return emb / np.linalg.norm(emb)

def image_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur_score = filters.laplace(img).var()
    return min(blur_score / 500, 1.0)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Face Authentication API (Embedding-Only, Phase 7)",
    description="Stores only face embeddings, not images",
    version="1.0"
)

# =========================
# ENROLL API
# =========================
@app.post("/enroll")
async def enroll(user_id: str, image: UploadFile = File(...)):
    temp_path = f"{UPLOAD_DIR}/{image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        emb = get_embedding(temp_path)
        vector_id = index.ntotal
        index.add(emb.reshape(1, -1))

        user_map.setdefault(user_id, []).append(vector_id)

        faiss.write_index(index, DB_PATH)
        pickle.dump(user_map, open(MAP_PATH, "wb"))

        return {
            "status": "enrolled",
            "user_id": user_id,
            "vector_id": vector_id
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # 🔒 image deleted immediately

# =========================
# AUTHENTICATE API
# =========================
@app.post("/authenticate")
async def authenticate(image: UploadFile = File(...)):
    temp_path = f"{UPLOAD_DIR}/{image.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    try:
        emb = get_embedding(temp_path)
        scores, _ = index.search(emb.reshape(1, -1), 1)

        similarity = float(scores[0][0])        # 🔥 convert to Python float
        quality = float(image_quality(temp_path))
        session_confidence = float(
            SIMILARITY_WEIGHT * similarity +
            QUALITY_WEIGHT * quality
        )

        authenticated = bool(session_confidence >= THRESHOLD)  # 🔥 convert to Python bool

        return {
            "authenticated": authenticated,
            "similarity_score": round(similarity, 3),
            "image_quality": round(quality, 3),
            "session_confidence": round(session_confidence, 3),
            "threshold": float(THRESHOLD)
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
