🚗 License Plate Detection & OCR System










📌 Overview

This project implements an end-to-end License Plate Recognition (LPR) system using:

🔍 YOLO (Ultralytics) for license plate detection
🔢 Tesseract OCR for extracting numeric text
🧠 Smart preprocessing techniques to boost OCR accuracy

It takes an image as input, detects license plates, extracts the text, and outputs the final annotated image.

✨ Features
🚘 Custom-trained YOLO model for plate detection
🔍 High-accuracy OCR pipeline (optimized for digits)
🧪 Debug image generation for tuning
⚡ Fast and lightweight processing
🧩 Modular and easy to extend
🛠️ Tech Stack
Tool	Purpose
Python	Core programming language
OpenCV	Image processing
Ultralytics YOLO	Object detection
Tesseract OCR	Text recognition
NumPy	Numerical operations
📂 Project Structure
.
├── license_plate_detector.pt   # YOLO trained model
├── car2.jpg                    # Input image
├── result.jpg                  # Final output
├── debug_full_plate.jpg        # Debug (full plate)
├── debug_left.jpg              # Debug (left segment)
├── debug_right.jpg             # Debug (right segment)
├── crop_debug.jpg              # Cropped plate
└── main.py                     # Main script
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/yourusername/license-plate-ocr.git
cd license-plate-ocr
2️⃣ Install Dependencies
pip install ultralytics opencv-python pytesseract numpy
3️⃣ Install Tesseract

macOS

brew install tesseract

Ubuntu

sudo apt install tesseract-ocr

Windows
Download: https://github.com/tesseract-ocr/tesseract

⚙️ Configuration

Update paths in main.py:

IMAGE_PATH = "path/to/image.jpg"
MODEL_PATH = "license_plate_detector.pt"

If Tesseract is not in PATH:

pytesseract.pytesseract.tesseract_cmd = "/path/to/tesseract"
▶️ Usage

Run the project:

python main.py
🔍 How It Works
1. Detection

YOLO detects license plate bounding boxes in the image.

2. Preprocessing
Convert to grayscale
Resize (×4 scaling)
Gaussian blur
Binary threshold
3. OCR Strategy
Try OCR on full plate
If weak result → split plate:
Left (70%)
Right (30%)
Merge results
4. Validation
Accept only results with ≥ 7 digits
🧪 Debug Outputs

These files help improve accuracy:

debug_full_plate.jpg
debug_left.jpg
debug_right.jpg
crop_debug.jpg
⚠️ Limitations
🔢 Currently supports numeric plates only
📷 Sensitive to lighting & image quality
🧠 Requires trained YOLO model
🚀 Future Improvements
🔤 Support alphanumeric plates
🎥 Real-time video detection
🌍 Region-specific plate formats
🧠 Advanced preprocessing (adaptive thresholding, deep OCR)
📄 License

This project is licensed under the MIT License.

👨‍💻 Author

Otmane Rachedi Abderrhmane
🚀 CEO — Cyberlymph

⭐ Support

If you like this project:

⭐ Star the repository
🍴 Fork it
🧠 Contribute improvements