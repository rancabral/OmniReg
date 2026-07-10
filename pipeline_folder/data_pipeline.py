"""OmniReg Data Pipeline - Image Preprocessing & Model Inference (Cross-Platform: PC & Raspberry Pi)"""
import cv2
import numpy as np
from pathlib import Path
import time
import platform

# Dual-platform TFLite import (graceful fallback)
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        from tensorflow.lite.python.lite import Interpreter
        TFLITE_SOURCE = "tensorflow"
    except ImportError:
        TFLITE_SOURCE = "not_installed"
        pass  # Model loading will gracefully fail with informative message


def preprocess_image(image_path, target_width=256, target_height=64, return_metadata=False):
    """
    Preprocess image for both OCR display quality and model input.
    
    Robust preprocessing steps:
    1. Dynamic Contrast Normalization (Adaptive CLAHE): Flatten local lighting gradients
    2. High-Pass Stroke Sharpening: Crisp up character boundaries with custom kernel
    3. Universal Aspect-Ratio Preserving Scale: Fit proportionally without distortion
    4. Standard Padded Background Canvas: White background with centered equation
    5. Clean Otsu Binarization: Standard thresholding (no inversion)
    
    Returns:
        default -> (model_binary, display_binary)
        return_metadata=True -> (model_binary, display_binary, metadata)
        metadata keys: equation_crop_gray, equation_bbox, full_binary
    """
    # Load image in grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    # Step 1: Dynamic Contrast Normalization (Adaptive CLAHE)
    # Flattens local phone-camera lighting gradients and shadows.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)
    
    # Step 2: High-Pass Stroke Sharpening
    # Crisp up character boundaries while preserving stroke structure.
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)
    # Clip to valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Step 3: Full-resolution binary for display/OCR (do not resize to avoid blur)
    _, full_binary = cv2.threshold(
        sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 4: Find tight equation bounding box from text mask
    text_mask = 255 - full_binary  # text becomes white on black mask
    coords = cv2.findNonZero(text_mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(2, int(0.02 * max(w, h)))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(full_binary.shape[1], x + w + pad)
        y1 = min(full_binary.shape[0], y + h + pad)
    else:
        # Fallback: use full image if no foreground was detected.
        x0, y0 = 0, 0
        x1, y1 = full_binary.shape[1], full_binary.shape[0]

    equation_crop_gray = img[y0:y1, x0:x1]
    display_binary = full_binary[y0:y1, x0:x1]
    
    # Step 5: Build model-sized binary while preserving equation aspect ratio
    h, w = display_binary.shape[:2]
    aspect_ratio = w / h
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Image is wider: width is limiting factor
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
        interpolation = cv2.INTER_AREA if new_w < w else cv2.INTER_NEAREST
    else:
        # Image is taller: height is limiting factor
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
        interpolation = cv2.INTER_AREA if new_h < h else cv2.INTER_NEAREST
    
    resized = cv2.resize(display_binary, (new_w, new_h), interpolation=interpolation)
    
    # Initialize with pure white, center scaled equation, pad edges with white.
    canvas = np.ones((target_height, target_width), dtype=np.uint8) * 255
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    # Ensure model input remains strict binary.
    _, model_binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if return_metadata:
        metadata = {
            "equation_crop_gray": equation_crop_gray,
            "equation_bbox": (x0, y0, x1, y1),
            "full_binary": full_binary,
        }
        return model_binary, display_binary, metadata

    return model_binary, display_binary


def infer_equation(image_path, model_path=None):
    """
    Run inference on preprocessed image
    Cross-platform: Works on PC and Raspberry Pi
    
    Returns: (decoded_equation, inference_time_ms, display_binary_image)
    """
    result = infer_equation_response(image_path=image_path, model_path=model_path)
    return result["decoded_equation"], result["inference_time_ms"], result["preprocessed_binary"]


def infer_equation_response(image_path, model_path=None):
    """
    Unified backend response mapping for UI/API integration.

    Returns dict:
        {
            "ocr_output": str,
            "decoded_equation": str,
            "decoded_equation_latex": str,
            "strict_match": bool,
            "inference_time_ms": float,
            "preprocessed_binary": np.ndarray | None,
        }
    """
    start_time = time.time()

    try:
        model_binary, display_binary, meta = preprocess_image(image_path, return_metadata=True)
    except Exception as e:
        return {
            "ocr_output": f"Error preprocessing: {str(e)[:30]}",
            "decoded_equation": f"Error preprocessing: {str(e)[:30]}",
            "decoded_equation_latex": "",
            "strict_match": False,
            "inference_time_ms": 0.0,
            "preprocessed_binary": None,
        }

    # Keep TFLite invocation path active for timing/compatibility even when OCR is authoritative.
    try:
        if model_path is None:
            BASE_DIR = Path(__file__).resolve().parent.parent
            model_path = BASE_DIR / "models" / "equation_model.tflite"

        if Interpreter is not None and Path(model_path).exists():
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            input_tensor = model_binary.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=(0, 3))
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
    except Exception:
        # Non-fatal: OCR path still returns a valid mapping payload.
        pass

    from nlp_folder.text_processor import ocr_extract, build_equation_mapping

    ocr_text = ocr_extract(meta.get("equation_crop_gray"))
    mapping = build_equation_mapping(ocr_text)
    elapsed_ms = (time.time() - start_time) * 1000

    return {
        "ocr_output": mapping["ocr_raw"],
        "decoded_equation": mapping["decoded_plain"],
        "decoded_equation_latex": mapping["decoded_latex"],
        "strict_match": mapping["is_strict_match"],
        "inference_time_ms": elapsed_ms,
        "preprocessed_binary": display_binary,
    }


def get_system_info():
    """Return system info for debugging"""
    return {
        'platform': platform.system(),
        'platform_version': platform.release(),
        'python_version': platform.python_version(),
        'tflite_source': TFLITE_SOURCE
    }
