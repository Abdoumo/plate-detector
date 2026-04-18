from ultralytics import YOLO
import cv2
import pytesseract
import re

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "/Users/naoufel/vision-env/car.jpeg"
MODEL_PATH = "license_plate_detector.pt"

# If needed on macOS, uncomment and set correct path:
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
    print("❌ ERROR: Image not found ->", IMAGE_PATH)
    exit()

print("✅ Image loaded:", image.shape)

# -----------------------------
# RUN DETECTION (IMPORTANT FIX)
# -----------------------------
results = model.predict(source=image, conf=0.25)[0]

# -----------------------------
# PROCESS DETECTIONS
# -----------------------------
found = False

for box in results.boxes:
    found = True

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Crop plate
    plate = image[y1:y2, x1:x2]

    if plate.size == 0:
        continue

    # -----------------------------
    # PREPROCESS FOR OCR
    # -----------------------------
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Resize for better OCR
    thresh = cv2.resize(thresh, None, fx=2, fy=2)

    # -----------------------------
    # OCR
    # -----------------------------
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)

    text = text.strip().replace(" ", "").replace("\n", "")

    # -----------------------------
    # OPTIONAL: Italian format check
    # -----------------------------
    match = re.findall(r"[A-Z]{2}\d{3}[A-Z]{2}", text)

    print("\n🔎 Raw OCR:", text)
    print("🚗 Valid plate:", match)

    # -----------------------------
    # DRAW BOX
    # -----------------------------
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        text if text else "PLATE",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# -----------------------------
# SAVE RESULT
# -----------------------------
if found:
    cv2.imwrite("result.jpg", image)
    print("\n✅ Saved: result.jpg")
else:
    print("\n⚠️ No license plates detected")
    cv2.imwrite("result.jpg", image)
