"""OmniReg NLP Text Processor - NLTK Tokenization & EasyOCR"""
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tokenize.regexp import RegexpTokenizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    RegexpTokenizer = None

import re
import numpy as np
import os

try:
    import easyocr
except ImportError:
    easyocr = None


# Download NLTK data if available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass


def tokenize_equation(equation_text):
    """
    Tokenize equation text while preserving mathematical symbols
    Returns: list of tokens
    """
    if not equation_text or "Error" in equation_text or "not found" in equation_text:
        return []
    
    if not NLTK_AVAILABLE or RegexpTokenizer is None:
        # Fallback: simple regex split
        tokens = re.findall(r'\w+|[+\-=\*/\(\)\[\]\{\}]', equation_text)
        return tokens if tokens else []
    
    # Custom tokenizer that preserves mathematical symbols (+, -, =, (), [], etc)
    tokenizer = RegexpTokenizer(r'\w+|[+\-=\*/\(\)\[\]\{\}]')
    tokens = tokenizer.tokenize(equation_text)
    
    return tokens if tokens else []


def normalize_equation_text(text):
    """
    Normalize OCR output into a cleaner math-equation string.
    """
    if not text:
        return "(No text detected)"

    cleaned = str(text)

    # Common Unicode and OCR symbol normalization.
    substitutions = {
        '−': '-',
        '–': '-',
        '—': '-',
        '×': '*',
        '·': '*',
        '÷': '/',
        '：': ':',
    }
    for src, dst in substitutions.items():
        cleaned = cleaned.replace(src, dst)

    # Collapse repeated whitespace and remove obvious OCR separators.
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.replace('|', 'I')

    # Keep math-relevant characters plus letters/numbers.
    filtered = re.sub(r'[^A-Za-z0-9\+\-\=\*\/\(\)\[\]\{\}\^\._:, ]', '', cleaned)
    filtered = re.sub(r'\s+', ' ', filtered).strip()

    return filtered if filtered else "(No text detected)"


def to_latex_equation(equation_text):
    """
    Convert normalized equation text to a LaTeX-ready expression.
    This formatter preserves the original OCR-decoded characters and order.
    """
    normalized = normalize_equation_text(equation_text)
    if not normalized or normalized == "(No text detected)":
        return ""

    latex = normalized

    # Convert simple exponent patterns a^b into a^{b} without changing symbol order.
    latex = re.sub(r'([A-Za-z0-9\)\]])\^([A-Za-z0-9]+)', r'\1^{\2}', latex)
    latex = latex.replace('*', r' \cdot ')
    latex = re.sub(r'\s+', ' ', latex).strip()
    return f"${latex}$"


def build_equation_mapping(ocr_text):
    """
    Build strict 1:1 OCR-to-decoded mapping payload.
    decoded_plain is intentionally identical to ocr_raw_normalized.
    """
    ocr_raw_normalized = normalize_equation_text(ocr_text)
    decoded_plain = ocr_raw_normalized
    decoded_latex = to_latex_equation(decoded_plain)
    return {
        "ocr_raw": ocr_raw_normalized,
        "decoded_plain": decoded_plain,
        "decoded_latex": decoded_latex,
        "is_strict_match": ocr_raw_normalized == decoded_plain,
    }


def refine_equation_with_gemini(equation_text):
    """
    Optionally refine OCR equation text with Gemini when GEMINI_API_KEY is set.
    Falls back to local normalized text when SDK/key is unavailable.
    """
    normalized = normalize_equation_text(equation_text)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or not normalized or normalized == "(No text detected)":
        return normalized

    prompt = (
        "You are correcting OCR text of a mathematical equation. "
        "Return only the corrected equation string. "
        "Do not add explanations. OCR text: " + normalized
    )

    # Preferred SDK: google-genai
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        candidate = getattr(response, "text", "")
        fixed = normalize_equation_text(candidate)
        if fixed and fixed != "(No text detected)":
            return fixed
    except Exception:
        pass

    # Legacy fallback: google-generativeai
    try:
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=api_key)
        model = genai_legacy.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        candidate = getattr(response, "text", "")
        fixed = normalize_equation_text(candidate)
        if fixed and fixed != "(No text detected)":
            return fixed
    except Exception:
        pass

    return normalized


def ocr_extract(image_input):
    """
    Extract text using EasyOCR (with equation-oriented cleanup).
    Uses local project cache to avoid Windows permission errors
    Accepts either image path or numpy image array.
    Returns: extracted text string
    """
    if easyocr is None:
        return "(EasyOCR not installed)"
    
    try:
        import os
        # Force local cache directories to avoid [WinError 5] Access Denied on protected Windows paths
        local_cache = os.path.join(os.path.dirname(__file__), "easyocr_cache")
        os.makedirs(local_cache, exist_ok=True)
        
        # Set both model and user network directories to local project folder
        reader = easyocr.Reader(
            ['en'], 
            gpu=False, 
            verbose=False, 
            model_storage_directory=local_cache,
            user_network_directory=local_cache,
            download_enabled=True
        )

        read_source = image_input
        if isinstance(image_input, np.ndarray):
            read_source = image_input
        else:
            read_source = str(image_input)

        # detail=1 gives confidence scores and better ordering control.
        result = reader.readtext(
            read_source,
            detail=1,
            paragraph=False,
            decoder='beamsearch'
        )

        if not result:
            return "(No text detected)"

        # Sort left-to-right using bounding box minimum x value.
        sorted_result = sorted(
            result,
            key=lambda item: min(point[0] for point in item[0])
        )

        # Keep low-confidence text only if everything is low confidence.
        confident_tokens = [txt for _, txt, conf in sorted_result if conf >= 0.15 and str(txt).strip()]
        if confident_tokens:
            text = " ".join(confident_tokens)
        else:
            text = " ".join(str(txt) for _, txt, _ in sorted_result if str(txt).strip())

        return normalize_equation_text(text)
    except Exception as e:
        error_msg = str(e)[:40]
        # Return graceful message for permission errors
        if "Access is denied" in error_msg or "WinError 5" in error_msg:
            return "(EasyOCR cache permission issue - running with limited model access)"
        return f"OCR Error: {error_msg}"
