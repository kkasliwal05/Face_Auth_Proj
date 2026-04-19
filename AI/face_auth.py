import os
import cv2
import faiss
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from skimage import filters

# =============================
# CONFIG
# =============================
DB_PATH = "face_db.index"
MAP_PATH = "user_map.pkl"
VECTOR_DIM = 512
SIM_THRESHOLD = 0.6

# =============================
# LOAD ARCFACE MODEL
# =============================
print("[INFO] Loading ArcFace model...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


# =============================
# LOAD / CREATE FAISS DB
# =============================
if os.path.exists(DB_PATH):
    print("[INFO] Loading existing face database...")
    index = faiss.read_index(DB_PATH)
    with open(MAP_PATH, "rb") as f:
        user_map = pickle.load(f)
else:
    print("[INFO] Creating new face database...")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    user_map = {}


# =============================
# FACE → 512D VECTOR
# =============================
def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")

    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")


# =============================
# ENROLL USER
# =============================
def enroll_user(user_id, image_paths):
    vectors = []

    for path in image_paths:
        emb = get_embedding(path)
        vectors.append(emb)

    vectors = np.vstack(vectors)
    start_id = index.ntotal
    index.add(vectors)

    user_map[user_id] = list(range(start_id, start_id + len(vectors)))

    faiss.write_index(index, DB_PATH)
    with open(MAP_PATH, "wb") as f:
        pickle.dump(user_map, f)

    print(f"[SUCCESS] {user_id} enrolled with {len(vectors)} images")


# =============================
# IMAGE QUALITY SCORE
# =============================
def image_quality(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur_score = filters.laplace(img).var()
    return min(blur_score / 500, 1.0)


# =============================
# LIVENESS SCORE (Placeholder)
# =============================
def liveness_score():
    return 0.9   # assume live


# =============================
# AUTHENTICATION
# =============================
def authenticate(image_path):
    query = get_embedding(image_path).reshape(1, -1)
    scores, ids = index.search(query, 5)

    best_score = float(scores[0][0])
    best_id = ids[0][0]

    matched_user = None
    for user, id_list in user_map.items():
        if best_id in id_list:
            matched_user = user
            break

    if matched_user is None:
        return None, best_score

    quality = image_quality(image_path)
    live = liveness_score()

    session_confidence = (
        0.6 * best_score +
        0.2 * live +
        0.2 * quality
    )

    return matched_user, session_confidence


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    # ===== ENROLL =====
    enroll_user(
        user_id="khushi",
        image_paths=[
            "enroll/khushi1.jpg",
            "enroll/khushi2.jpg",
            "enroll/khushi3.jpg"
        ]
    )

    # ===== AUTHENTICATE =====
    user, confidence = authenticate("login/test.jpg")

    if user and confidence > SIM_THRESHOLD:
        print(f"AUTH SUCCESS: {user} | confidence={confidence:.2f}")
    else:
        print("AUTH FAILED")
