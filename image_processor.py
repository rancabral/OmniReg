import cv2
import numpy as np

def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check the file path.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Noise removal
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return processed
