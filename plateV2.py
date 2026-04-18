from ultralytics import YOLO
import cv2
import pytesseract
import re
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "/Users/Cyberlymph/vision-env/car2.jpg"
MODEL_PATH = "license_plate_detector.pt"

# If needed on macOS:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Image not found")
    exit()

print("✅ Image loaded:", image.shape)

# -----------------------------
# OCR FUNCTION
# -----------------------------
def ocr_segment(img):
    config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(img, config=config)
    return re.sub(r'[^0-9]', '', text)

# -----------------------------
# PROCESS PLATE (KEY PART)
# -----------------------------
def read_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # upscale (keep this)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # blur FIRST (not after threshold)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # LIGHT threshold (not aggressive)
    _, gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    cv2.imwrite("debug_full_plate.jpg", gray)

    # 🔥 IMPORTANT: try OCR on FULL plate first
    full_text = ocr_segment(gray)
    print("FULL OCR:", full_text)

    if len(full_text) >= 7:
        return full_text

    # -----------------------------
    # fallback → smart split
    # -----------------------------
    h, w = gray.shape

    left = gray[:, 0:int(w * 0.7)]
    right = gray[:, int(w * 0.7):w]

    cv2.imwrite("debug_left.jpg", left)
    cv2.imwrite("debug_right.jpg", right)

    left_text = ocr_segment(left)
    right_text = ocr_segment(right)

    print("LEFT :", left_text)
    print("RIGHT:", right_text)

    return f"{left_text} {right_text}".strip()

# -----------------------------
# DETECTION
# -----------------------------
results = model.predict(source=image, conf=0.25)[0]

final_plate = None

# -----------------------------
# PROCESS BOXES
# -----------------------------
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        continue

    cv2.imwrite("crop_debug.jpg", plate)

    text = read_plate(plate)

    print("🔎 OCR RESULT:", text)

    # accept if reasonable length
    clean_text = text.replace(" ", "")
    if len(clean_text) >= 7:
        final_plate = text

    # draw box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# -----------------------------
# OUTPUT
# -----------------------------
if final_plate:
    cv2.putText(
        image,
        final_plate,
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 255, 0),
        4
    )
    print("🚗 FINAL PLATE:", final_plate)
else:
    cv2.putText(
        image,
        "NO PLATE",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4
    )
    print("⚠️ No valid plate detected")

# -----------------------------
# SAVE RESULT
# -----------------------------
cv2.imwrite("result.jpg", image)
print("✅ Saved: result.jpg")
