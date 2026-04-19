import os
from tkinter import Tk, filedialog
from rembg import remove
from PIL import Image
from datetime import datetime

# ---------------- USER IMAGE INPUT ----------------
def select_image():
    Tk().withdraw()
    return filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

image_path = select_image()

if not image_path:
    print(" No image selected. Exiting.")
    exit()

print(f"Image selected: {image_path}")

# ---------------- CREATE UNIQUE OUTPUT FOLDER ----------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_folder = os.path.join("output", f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# ---------------- LOAD IMAGE ----------------
input_image = Image.open(image_path).convert("RGBA")

# Save original
original_path = os.path.join(run_folder, "original.png")
input_image.convert("RGB").save(original_path)

# ---------------- BACKGROUND REMOVAL ----------------
removed_bg = remove(input_image)

bg_removed_path = os.path.join(run_folder, "bg_removed.png")
removed_bg.save(bg_removed_path)

# ---------------- PLACE ON WHITE BACKGROUND ----------------
white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
final_image = Image.alpha_composite(white_bg, removed_bg).convert("RGB")

final_path = os.path.join(run_folder, "bg_removed_white.png")
final_image.save(final_path)

# ---------------- FINAL STATUS ----------------
print("All outputs saved successfully")
print("Output folder:", run_folder)
print("STAGE 1 COMPLETED SUCCESSFULLY")
