import os
import pickle
import time
import warnings
import random
import numpy as np
import faiss
import cv2
import mediapipe as mp
from insightface.app import FaceAnalysis

warnings.filterwarnings("ignore")

# --- HELPER CLASS ---
class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_metrics(self, image):
        """Returns (ear, mar, yaw) efficiently."""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return None, None, None
        
        lm = results.multi_face_landmarks[0].landmark
        
        # EAR (Left Eye)
        v_dist = np.sqrt((lm[159].x - lm[145].x)**2 + (lm[159].y - lm[145].y)**2)
        h_dist = np.sqrt((lm[133].x - lm[33].x)**2 + (lm[133].y - lm[33].y)**2)
        ear = v_dist / h_dist if h_dist > 0 else 0

        # MAR (Smile)
        mouth_w = np.sqrt((lm[61].x - lm[291].x)**2 + (lm[61].y - lm[291].y)**2)
        face_w = np.sqrt((lm[234].x - lm[454].x)**2 + (lm[234].y - lm[454].y)**2)
        mar = mouth_w / face_w if face_w > 0 else 0

        # Yaw (Head Turn)
        nose_x = lm[1].x
        cheek_l_x = lm[33].x
        cheek_r_x = lm[263].x
        face_width = cheek_r_x - cheek_l_x
        if face_width == 0: return ear, mar, 0
        
        ratio = (nose_x - cheek_l_x) / face_width
        yaw = (ratio - 0.5) * 180 
        
        return ear, mar, yaw

# --- MAIN SYSTEM ---
class FaceAuthSystem:
    def __init__(self, db_path="face_db.index", map_path="user_map.pkl", state_path="security_state.pkl"):
        self.db_path = db_path
        self.map_path = map_path
        self.state_path = state_path 
        self.liveness_detector = FaceMeshDetector()
        
        # --- PROGRESSIVE SECURITY CONFIG ---
        self.failed_attempts = 0
        self.lockout_until = 0
        
        # Load Security State
        if os.path.exists(self.state_path):
            with open(self.state_path, 'rb') as f:
                state = pickle.load(f)
                self.failed_attempts = state.get('failures', 0)
                self.lockout_until = state.get('lockout', 0)
        
        # --- STARTUP STATUS ---
        print("\n" + "="*30)
        print("   SYSTEM STARTUP   ")
        # Check if Hard Locked (10+ fails)
        if self.failed_attempts >= 10:
             print("   STATUS: HARD LOCKED (Manual Review Required)")
        elif time.time() < self.lockout_until:
            print(f"   STATUS: TEMPORARY LOCK ({int(self.lockout_until - time.time())}s remaining)")
        else:
            print(f"   STATUS: READY (Failures: {self.failed_attempts})")
        print("="*30 + "\n")

        print("Loading ArcFace model (Lightweight Mode)...")
        # Using buffalo_s for optimization
        self.app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.dimension = 512
        
        if os.path.exists(db_path):
            self.index = faiss.read_index(db_path)
            with open(map_path, 'rb') as f: self.user_map = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.user_map = {} 

    def save_security_state(self):
        state = {
            'failures': self.failed_attempts,
            'lockout': self.lockout_until
        }
        with open(self.state_path, 'wb') as f:
            pickle.dump(state, f)

    def get_embedding(self, img_array):
        faces = self.app.get(img_array)
        if not faces: return None, 0.0
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[-1]
        embedding = face.embedding
        vector = np.array([embedding], dtype='float32')
        faiss.normalize_L2(vector)
        return vector[0], face.det_score

    def enroll_user(self, user_name, mode='camera'):
        if mode == 'camera':
            cap = cv2.VideoCapture(0)
            img_list = []
            count = 0
            print("Press SPACE to capture 3 angles (Front, Left, Right)")
            while count < 3:
                ret, frame = cap.read()
                cv2.imshow("Enroll", frame)
                if cv2.waitKey(1) & 0xFF == 32: # Spacebar
                    img_list.append(frame)
                    count += 1
                    print(f"Captured {count}/3")
                    time.sleep(0.5)
            cap.release(); cv2.destroyAllWindows()
        else:
            return 
            
        embeddings = []
        for img in img_list:
            emb, _ = self.get_embedding(img)
            if emb is not None: embeddings.append(emb)
            
        if not embeddings: return False
        
        avg_emb = np.mean(embeddings, axis=0)
        vector = np.array([avg_emb], dtype='float32')
        faiss.normalize_L2(vector)
        self.index.add(vector)
        self.user_map[self.index.ntotal - 1] = user_name 
        faiss.write_index(self.index, self.db_path)
        with open(self.map_path, 'wb') as f: pickle.dump(self.user_map, f)
        print(f"SUCCESS: Enrolled {user_name}")

    # --- RANDOMIZED LIVENESS CHECK ---
    def verify_liveness_video(self):
        cap = cv2.VideoCapture(0)
        
        challenges = ["BLINK", "SMILE", "TURN"]
        random.shuffle(challenges)
        
        print("\n--- ACTIVE PROOF OF LIFE ---")
        print(f"Randomized Sequence: {' -> '.join(challenges)} -> CENTER")
        
        start_time = time.time()
        step_idx = 0
        blink_closed = False
        
        best_frame = None
        best_quality_score = 0.0
        
        BLINK_THRESH = 0.30       
        SMILE_THRESH = 0.45       
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            ear, smile_ratio, yaw = self.liveness_detector.get_metrics(frame)
            if ear is None: continue

            if abs(yaw) < 10 and step_idx < 3 and blur_score > best_quality_score:
                best_quality_score = blur_score
                best_frame = frame.copy()

            if time.time() - start_time > 60:
                cap.release(); cv2.destroyAllWindows()
                return False, None

            status = "Align Face"
            
            if step_idx < 3:
                current_challenge = challenges[step_idx]
                status = f"Step {step_idx + 1}: {current_challenge} NOW!"
                
                if current_challenge == "BLINK":
                    if not blink_closed:
                        if ear < BLINK_THRESH: blink_closed = True
                    else:
                        if ear > BLINK_THRESH + 0.05:
                            print(f">> {current_challenge} Verified!")
                            step_idx += 1; blink_closed = False 

                elif current_challenge == "SMILE":
                    bar_len = int((smile_ratio - 0.35) * 400)
                    cv2.rectangle(frame, (20, 400), (20 + max(0, min(bar_len, 200)), 420), (0, 255, 255), -1)
                    if smile_ratio > SMILE_THRESH:
                        print(f">> {current_challenge} Verified!")
                        step_idx += 1; time.sleep(0.5)

                elif current_challenge == "TURN":
                    if abs(yaw) > 12:
                        print(f">> {current_challenge} Verified!")
                        step_idx += 1

            else:
                status = "Final Step: Look at Camera"
                if abs(yaw) < 8:
                    cv2.putText(frame, "VERIFIED", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow("Check", frame); cv2.waitKey(500)
                    cap.release(); cv2.destroyAllWindows()
                    return True, (best_frame if best_frame is not None else frame)

            color = (0, 255, 0) if step_idx < 3 else (0, 255, 255)
            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Check", frame)
            
            if cv2.waitKey(1) == 27: break
        
        cap.release(); cv2.destroyAllWindows()
        return False, None

    # --- AUTHENTICATE ---
    def authenticate(self):
        # 1. CHECK HARD LOCK
        if self.failed_attempts >= 10:
            print("\n" + "="*30)
            print("ACCOUNT SUSPENDED")
            print("Maximum security failures reached.")
            print("Please contact administrator for manual review.")
            print("="*30 + "\n")
            return

        # 2. CHECK TEMP LOCK
        if time.time() < self.lockout_until:
            print("\n" + "="*30)
            print("Authentication failed.")
            print("Account temporarily locked.")
            print(f"Please try again in {int(self.lockout_until - time.time())} seconds.")
            print("="*30 + "\n")
            return

        method = input("Authenticate via Camera (C) or Upload (U)? ").upper()
        
        try:
            if method == 'C':
                liveness, frame = self.verify_liveness_video()
                if not liveness: raise ValueError("Liveness Fail")
                emb, quality = self.get_embedding(frame)
            else:
                p = input("Path: ").strip().strip("'")
                img = cv2.imread(p)
                if img is None: raise ValueError("Load Fail")
                emb, quality = self.get_embedding(img)

            if emb is None: raise ValueError("No Face")

            dists, ids = self.index.search(np.array([emb]), k=1)
            sim_score = dists[0][0]
            session_conf = (sim_score * 0.7) + (quality * 0.3)
            
            auth_strength = "WEAK"
            if session_conf >= 0.85: auth_strength = "STRONG"
            elif session_conf >= 0.70: auth_strength = "MEDIUM"

            print(f"\n[DEBUG] Sim: {sim_score:.2f} | Qual: {quality:.2f} | Conf: {session_conf:.2f} ({auth_strength})")

            # STRICT THRESHOLDS
            if sim_score < 0.60: raise ValueError(f"Sim Low ({sim_score:.2f})")
            if session_conf < 0.70: raise ValueError(f"Conf Low ({session_conf:.2f})")

            # SUCCESS: RESET ALL COUNTERS
            print("\n" + "="*30)
            print("   ACCESS GRANTED   ")
            print(f"   User: {self.user_map.get(ids[0][0], 'Unknown')}")
            print(f"   Auth Strength: {auth_strength}")
            print("="*30 + "\n")
            
            self.failed_attempts = 0 
            self.lockout_until = 0
            self.save_security_state()

        except ValueError as e:
            # FAILURE LOGIC (PROGRESSIVE)
            self.failed_attempts += 1
            
            # --- TIER 3: HARD LOCK (10+) ---
            if self.failed_attempts >= 10:
                print("\n" + "="*30)
                print("   ACCESS DENIED   ")
                print("   [CRITICAL] Security Alert")
                print("   10/10 Failures reached.")
                print("   ACCOUNT PERMANENTLY LOCKED.")
                print("="*30 + "\n")
                self.lockout_until = 9999999999 # Forever
            
            # --- TIER 2: LONG LOCK (6) ---
            elif self.failed_attempts == 6:
                self.lockout_until = time.time() + 120 # 2 Minutes
                print("\n" + "="*30)
                print("   ACCESS DENIED   ")
                print("   [WARNING] Suspicious activity detected.")
                print("   Account locked for 2 MINUTES.")
                print("="*30 + "\n")
            
            # --- TIER 1: SHORT LOCK (3) ---
            elif self.failed_attempts == 3:
                self.lockout_until = time.time() + 30 # 30 Seconds
                print("\n" + "="*30)
                print("   ACCESS DENIED   ")
                print("   Authentication failed.")
                print("   Account locked for 30 SECONDS.")
                print("="*30 + "\n")
            
            # --- WARNINGS ---
            else:
                remaining_tier = 10 - self.failed_attempts
                print("\n" + "="*30)
                print("   ACCESS DENIED   ")
                print("   Authentication Failed")
                print("-" * 30)
                print(f"   [WARNING] Failure {self.failed_attempts}/10")
                if self.failed_attempts < 3:
                    print(f"   (Lockout in {3 - self.failed_attempts} attempts)")
                elif self.failed_attempts < 6:
                    print(f"   (Long Lockout in {6 - self.failed_attempts} attempts)")
                else:
                    print(f"   (PERMANENT BLOCK in {10 - self.failed_attempts} attempts)")
                print("="*30 + "\n")
            
            self.save_security_state()

if __name__ == "__main__":
    s = FaceAuthSystem()
    a = input("Type 'E' to Enroll, 'A' to Authenticate: ").upper()
    if a == 'E': s.enroll_user(input("Name: "))
    elif a == 'A': s.authenticate()