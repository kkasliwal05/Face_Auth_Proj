import os
import cv2
import faiss
import pickle
import numpy as np
from insightface.app import FaceAnalysis
from skimage import filters

# =========================
# CONFIG
# =========================
DB_PATH = "face_db.index"
MAP_PATH = "user_map.pkl"
VECTOR_DIM = 512
THRESHOLD = 0.6

# =========================
# LOAD ARCFACE MODEL
# =========================
print("[INFO] Loading ArcFace model...")
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# LOAD / CREATE DATABASE
# =========================
if os.path.exists(DB_PATH) and os.path.exists(MAP_PATH):
    index = faiss.read_index(DB_PATH)
    with open(MAP_PATH, "rb") as f:
        user_map = pickle.load(f)
else:
    index = faiss.IndexFlatIP(VECTOR_DIM)
    user_map = {}

# =========================
# FACE → EMBEDDING
# =========================
def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not readable")

    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected")

    emb = faces[0].embedding
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")

# =========================
# IMAGE QUALITY
# =========================
def image_quality(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = filters.laplace(img).var()
    return min(blur / 500, 1.0)

# =========================
# ENROLL PERSON (MULTI-ANGLE)
# =========================
def enroll_person(person_name, image_paths):
    vectors = []
    records = []

    for img_path in image_paths:
        try:
            angle = os.path.splitext(os.path.basename(img_path))[0]
            emb = get_embedding(img_path)

            vectors.append(emb)
            records.append(angle)

            print(f"✔ Face detected: {img_path}")

        except Exception as e:
            print(f"❌ Skipped {img_path} | {e}")

    if len(vectors) == 0:
        print("❌ No valid images found. Enrollment aborted.")
        return

    vectors = np.vstack(vectors)
    start_id = index.ntotal
    index.add(vectors)

    user_map.setdefault(person_name, [])
    for i, angle in enumerate(records):
        user_map[person_name].append({
            "id": start_id + i,
            "angle": f"{person_name}_{angle}"
        })

    faiss.write_index(index, DB_PATH)
    with open(MAP_PATH, "wb") as f:
        pickle.dump(user_map, f)

    print(f"Enrolled {person_name} with {len(vectors)} angles")

# =========================
# AUTHENTICATE
# =========================
def authenticate(img_path):
    query = get_embedding(img_path).reshape(1, -1)
    scores, ids = index.search(query, 3)

    best_score = float(scores[0][0])
    best_id = int(ids[0][0])

    for person, records in user_map.items():
        for r in records:

            # ONLY accept dict format
            if isinstance(r, dict):
                if r.get("id") == best_id:
                    quality = image_quality(img_path)
                    confidence = 0.7 * best_score + 0.3 * quality
                    return person, r.get("angle"), confidence

            # ignore everything else safely
            else:
                continue

    return None, None, 0

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    choice = input("Type 'E' to Enroll, 'A' to Authenticate: ").strip().upper()

    if choice == "E":
        person = input("Person Name: ").strip()
        folder = f"enroll/{person}"

        if not os.path.exists(folder):
            print(f" Folder not found: {folder}")
            print(" Create folder and add images")
            exit()

        images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        enroll_person(person, images)

    elif choice == "A":
        img = input("Path to login image: ").strip()
        person, angle, conf = authenticate(img)

        if person and conf > THRESHOLD:
            print("\n AUTH SUCCESS")
            print(f"Person: {person}")
            print(f"Matched Angle: {angle}")
            print(f"Confidence: {conf:.2f}")
        else:
            print("\n AUTH FAILED")
        
