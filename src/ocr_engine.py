import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = RapidOCR()
    return _ocr


def read_region(img, x1, y1, x2, y2) -> str:
    """Extract text from image region via OCR."""
    if img is None:
        return ""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return ""
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return ""

    # Upscale small regions for better OCR accuracy
    rh, rw = region.shape[:2]
    if rh < 40 or rw < 100:
        scale = max(2, 40 // max(rh, 1))
        region = cv2.resize(region, (rw * scale, rh * scale), interpolation=cv2.INTER_CUBIC)

    ocr = _get_ocr()
    result, _ = ocr(region)
    if not result:
        return ""

    # Concatenate all detected text boxes
    texts = []
    for item in result:
        if item and len(item) >= 2:
            text = item[1]
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            if text:
                texts.append(str(text).strip())
    return " ".join(texts).strip()


def read_region_lines(img, x1, y1, x2, y2) -> list:
    """Extract text lines from image region, returned as list of strings."""
    if img is None:
        return []
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return []
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        return []

    rh, rw = region.shape[:2]
    if rh < 40 or rw < 100:
        scale = max(2, 40 // max(rh, 1))
        region = cv2.resize(region, (rw * scale, rh * scale), interpolation=cv2.INTER_CUBIC)

    ocr = _get_ocr()
    result, _ = ocr(region)
    if not result:
        return []

    lines = []
    for item in result:
        if item and len(item) >= 2:
            text = item[1]
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            if text:
                lines.append(str(text).strip())
    return lines
