import os
import json
import cv2
import numpy as np
import requests
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ============================================================
# CONFIGURATION & GEMINI CLIENT SETUP
# ============================================================

CAMERA_URL = "http://172.16.44.76:8080/shot.jpg"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)


class PillDimension(BaseModel):
    pill_id: int = Field(description="Sequential pill index (1, 2, ...)")
    length_mm: float = Field(description="Major dimension/length in mm based on graph grid")
    width_mm: float = Field(description="Minor dimension/width in mm based on graph grid")

class GeminiGridAnalysis(BaseModel):
    pill_count: int = Field(description="Total count of visible whole or partial pills")
    pills: list[PillDimension] = Field(description="List of detected pills with dimensions")


# ============================================================
# CAMERA STREAM HANDLER
# ============================================================

def capture_image(camera_url=CAMERA_URL):
    try:
        response = requests.get(camera_url, timeout=8)
        response.raise_for_status()
        image_array = np.frombuffer(response.content, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[Camera Error]: {e}")
        return None


# ============================================================
# OPENCV: SHAPE & COLOR PROCESSING
# ============================================================

HUE_RANGES = {
    "red": [(0, 10), (170, 179)],
    "orange": [(11, 22)],
    "yellow": [(23, 35)],
    "green": [(36, 85)],
    "blue": [(86, 125)],
    "purple": [(126, 145)],
    "pink": [(146, 169)]
}

def normalize_contour(contour):
    if contour is None:
        return None
    contour = np.asarray(contour, dtype=np.int32)
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape((-1, 1, 2))
    return np.ascontiguousarray(contour, dtype=np.int32) if len(contour) >= 3 else None

def get_primary_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = [c for c in contours if cv2.contourArea(c) > 250]
    return max(valid, key=cv2.contourArea) if valid else None

def detect_colour(image, contour):
    contour = normalize_contour(contour)
    if contour is None:
        return "UNKNOWN"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    pixels = hsv[mask == 255]
    if pixels.size == 0:
        return "UNKNOWN"

    mean_h, mean_s, mean_v = pixels.mean(axis=0)

    # Achromatic check
    if mean_s < 35:
        if mean_v > 180:
            return "white"
        if mean_v < 50:
            return "black"
        return "grey"

    for name, ranges in HUE_RANGES.items():
        for lo, hi in ranges:
            if lo <= mean_h <= hi:
                return name
    return "unregistered"

def classify_shape(contour):
    contour = normalize_contour(contour)
    if contour is None:
        return "irregular"

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = (max(w, h) / min(w, h)) if min(w, h) > 0 else 0
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    extent = (area / (w * h)) if w * h > 0 else 0

    if extent > 0.85 and aspect_ratio < 1.3:
        return "rectangular"
    if circularity > 0.82 and aspect_ratio < 1.2:
        return "round"
    if aspect_ratio >= 1.55 and extent > 0.70:
        return "capsule"
    if aspect_ratio >= 1.2:
        return "oval"
    return "irregular"


# ============================================================
# GEMINI: COUNT & GRID-SCALE DETECTION
# ============================================================

def analyze_grid_and_count_with_gemini(image_bgr):
    rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    prompt = (
        "Analyze the provided image of medication placed on top of a graph grid background.\n"
        "1. Count how many individual pills are present.\n"
        "2. Using the millimeter graph grid background as the spatial reference, measure the major dimension (length_mm) "
        "and minor dimension (width_mm) in millimeters for each pill."
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, pil_image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiGridAnalysis,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[Gemini API Error]: {e}")
        return None


# ============================================================
# FLASK-FACING MAIN INTERFACE
# ============================================================

def analyze_medication(image=None):
    """
    Main function consumed by verify() in app.py.
    Returns JSON dictionary compatible with verification gates 3-8.
    """
    if image is None:
        image = capture_image(CAMERA_URL)

    if image is None:
        return {"status": "CAMERA_OFFLINE", "message": "Failed to read camera feed."}

    # 1. OpenCV Contour, Color & Shape Detection
    contour = get_primary_contour(image)
    if contour is None:
        return {"status": "PILL_NOT_FOUND", "message": "No pill contour isolated."}

    detected_color = detect_colour(image, contour)
    detected_shape = classify_shape(contour)

    # 2. Gemini API Grid-Scale Dimension & Pill Count Detection
    gemini_data = analyze_grid_and_count_with_gemini(image)
    if not gemini_data or not gemini_data.get("pills"):
        return {"status": "GRID_SCALE_ERROR", "message": "Gemini could not parse dimensions."}

    # Extract primary pill size (major length in mm) and count
    pill_count = gemini_data.get("pill_count", 1)
    primary_pill = gemini_data["pills"][0]
    detected_size_mm = float(primary_pill.get("length_mm", 0.0))
    detected_width_mm = float(primary_pill.get("width_mm", 0.0))

    # Reject if multiple pills are on the plate when single-dose verification is expected
    if pill_count > 1:
        return {
            "status": "MULTIPLE_PILLS_DETECTED",
            "pill_count": pill_count,
            "color": detected_color,
            "shape": detected_shape,
            "size_mm": detected_size_mm
        }

    # 3. Payload expected by app.py gates
    return {
        "status": "SUCCESS",
        "color": detected_color,
        "shape": detected_shape,
        "size_mm": detected_size_mm,
        "width_mm": detected_width_mm,
        "pill_count": pill_count
    }


if __name__ == "__main__":
    result = analyze_medication()
    print(json.dumps(result, indent=2))