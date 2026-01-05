import os
import pickle
import numpy as np
import faiss
import cv2
from insightface.app import FaceAnalysis

class FaceAuthSystem:
    def __init__(self, db_path="face_db.index", map_path="user_map.pkl"):
        self.db_path = db_path
        self.map_path = map_path

        print("Loading ArcFace model...")
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # UNCHANGED

        self.dimension = 512

        if os.path.exists(db_path):
            print(f"Loading existing database from {db_path}...")
            self.index = faiss.read_index(db_path)
            with open(map_path, "rb") as f:
                self.user_map = pickle.load(f)
        else:
            print("Creating new vector database...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.user_map = {}

    # =========================
    # FACE → EMBEDDING
    # =========================
    def get_embedding(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        faces = self.app.get(img)
        if not faces:
            return None, 0.0

        # Largest face selection (UNCHANGED LOGIC)
        face = max(
            faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
        )

        vector = np.asarray(face.embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vector)

        return vector[0], face.det_score

    # =========================
    # ENROLL USER
    # =========================
    def enroll_user(self, img_path, user_name):
        try:
            embedding, quality = self.get_embedding(img_path)
            if embedding is None:
                print(f"Failed to enroll {user_name}: No face detected.")
                return False

            self.index.add(embedding.reshape(1, -1))
            new_id = self.index.ntotal - 1
            self.user_map[new_id] = user_name

            faiss.write_index(self.index, self.db_path)
            with open(self.map_path, "wb") as f:
                pickle.dump(self.user_map, f)

            print(
                f"SUCCESS: Enrolled {user_name} "
                f"(ID: {new_id}, Quality: {quality:.2f})"
            )
            return True

        except Exception as e:
            print(f"Error enrolling user: {e}")
            return False

    # =========================
    # AUTHENTICATE USER
    # =========================
    def authenticate(self, img_path, threshold=0.4):
        embedding, quality = self.get_embedding(img_path)
        if embedding is None:
            return "No Face", 0.0, False

        distances, ids = self.index.search(embedding.reshape(1, -1), k=1)
        best_match_id = ids[0][0]
        similarity_score = distances[0][0]

        matched_user = self.user_map.get(best_match_id, "Unknown")

        final_score = (similarity_score * 0.7) + (quality * 0.3)

        is_verified = (similarity_score > threshold) and (quality > 0.5)

        return {
            "user": matched_user,
            "verified": is_verified,
            "similarity": similarity_score,
            "quality": quality,
            "final_score": final_score
        }

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    system = FaceAuthSystem()
    action = input("Type 'E' to Enroll, 'A' to Authenticate: ").upper()

    if action == "E":
        path = input("Path to image: ").strip().strip("'")
        name = input("User Name: ")
        system.enroll_user(path, name)

    elif action == "A":
        path = input("Path to test image: ").strip().strip("'")
        result = system.authenticate(path)

        print("\n--- AUTHENTICATION REPORT ---")
        print(f"User Identity: {result['user']}")
        print(f"Access Granted: {result['verified']}")
        print(f"Similarity Score: {result['similarity']:.4f} (Max 1.0)")
        print(f"Image Quality: {result['quality']:.4f}")
        print(f"Session Confidence: {result['final_score']:.4f}")
