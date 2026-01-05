import os
from tkinter import Tk, filedialog
from rembg import remove
from PIL import Image
from datetime import datetime

# ---------------- USER IMAGE INPUT (GUI - SAME AS BEFORE) ----------------
def select_image():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # bring dialog to front
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path

image_path = select_image()

if not image_path:
    print("No image selected. Exiting.")
    exit()

print(f"Image selected: {image_path}")

# ---------------- CREATE UNIQUE OUTPUT FOLDER ----------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_folder = os.path.join("output", f"run_{timestamp}")
os.makedirs(run_folder, exist_ok=True)

# ---------------- LOAD IMAGE & PROCESS ----------------
with Image.open(image_path).convert("RGBA") as input_image:

    # Save original
    original_path = os.path.join(run_folder, "original.png")
    input_image.convert("RGB").save(original_path)

    # Background removal
    removed_bg = remove(input_image)
    bg_removed_path = os.path.join(run_folder, "bg_removed.png")
    removed_bg.save(bg_removed_path)

    # Place on white background
    white_bg = Image.new("RGBA", removed_bg.size, (255, 255, 255, 255))
    final_image = Image.alpha_composite(white_bg, removed_bg).convert("RGB")
    final_path = os.path.join(run_folder, "bg_removed_white.png")
    final_image.save(final_path)

# ---------------- FINAL STATUS ----------------
print("All outputs saved successfully")
print("Output folder:", run_folder)
print("STAGE 1 COMPLETED SUCCESSFULLY")
