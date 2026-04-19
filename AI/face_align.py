import cv2
import os
import numpy as np
from glob import glob

# ---------------- CONFIG ----------------
FINAL_SIZE = 224        # High-res face size
BORDER_PAD = 8          # Soft border size (pixels)

# ---------------- FIND ALL RUN FOLDERS ----------------
run_folders = sorted(glob("output/run_*"))

if not run_folders:
    print("No run folders found. Run main.py first.")
    exit()

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- PROCESS EACH IMAGE ----------------
for run in run_folders:

    input_img = os.path.join(run, "bg_removed_white.png")
    if not os.path.exists(input_img):
        print(f"⚠ Skipping {run} (bg_removed_white.png not found)")
        continue

    img = cv2.imread(input_img)
    if img is None:
        print(f"⚠ Could not read image in {run}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(80, 80)   # ignore tiny / low-quality faces
    )

    if len(faces) == 0:
        print(f"No face detected in {run}")
        continue

    # Take the largest face (best for recognition)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    # ---------------- GENEROUS FACE CROP ----------------
    margin = int(0.40 * w)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    face_crop = img[y1:y2, x1:x2]

    # ---------------- HIGH QUALITY RESIZE ----------------
    face_resized = cv2.resize(
        face_crop,
        (FINAL_SIZE, FINAL_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    # ---------------- EDGE-GUIDED STRUCTURE ENHANCEMENT ----------------
    edges = cv2.Canny(face_resized, 80, 160)

    edge_mask = cv2.GaussianBlur(edges, (7, 7), 0)
    edge_mask = edge_mask / 255.0
    edge_mask = np.expand_dims(edge_mask, axis=2)

    blur = cv2.GaussianBlur(face_resized, (0, 0), 1.0)
    sharpened = cv2.addWeighted(face_resized, 1.2, blur, -0.2, 0)

    face_structured = (
        face_resized * (1 - edge_mask) +
        sharpened * edge_mask
    ).astype(np.uint8)

    # ---------------- BORDER HANDLING (VERY IMPORTANT) ----------------
    # Replicated padding (no artificial edges)
    face_padded = cv2.copyMakeBorder(
        face_structured,
        BORDER_PAD, BORDER_PAD, BORDER_PAD, BORDER_PAD,
        cv2.BORDER_REPLICATE
    )

    # Resize back to FINAL_SIZE
    face_padded = cv2.resize(
        face_padded,
        (FINAL_SIZE, FINAL_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Very light smoothing on borders
    face_final = cv2.GaussianBlur(face_padded, (3, 3), 0)

    # ---------------- SAVE OUTPUTS ----------------
    out_aligned = os.path.join(run, "face_aligned_224.png")
    out_structured = os.path.join(run, "face_structured.png")

    cv2.imwrite(out_aligned, face_resized)
    cv2.imwrite(out_structured, face_final)

    print(f"Processed: {run}")
    print("   ├─ face_aligned_224.png")
    print("   └─ face_structured.png")

print("Face alignment + structure + border handling completed")
