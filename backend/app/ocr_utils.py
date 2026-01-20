import io
from typing import Tuple
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover - handled at runtime
    pytesseract = None


class OCRError(Exception):
    """Raised when OCR fails."""


def extract_text_from_image(file_bytes: bytes) -> Tuple[str, str]:
    """Extract text from an image using Tesseract.

    Returns a tuple of (text, engine_used).
    Raises OCRError if pytesseract is not available.
    """
    if pytesseract is None:
        raise OCRError(
            "pytesseract is not installed. Install it and ensure the Tesseract binary is available on PATH."
        )

    try:
        image = Image.open(io.BytesIO(file_bytes))
    except Exception as exc:  # pragma: no cover - relies on runtime files
        raise OCRError(f"Failed to open image: {exc}") from exc

    try:
        text = pytesseract.image_to_string(image)
    except Exception as exc:  # pragma: no cover
        raise OCRError(f"Failed to run OCR: {exc}") from exc

    cleaned = text.strip()
    if not cleaned:
        raise OCRError("No text detected in image")

    return cleaned, "tesseract"
