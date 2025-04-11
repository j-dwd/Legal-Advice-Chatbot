from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

def process_multimodal_input(file):
    """Extract text from images or PDFs."""
    try:
        if file.type.startswith("image"):
            img = Image.open(file).convert("RGB")
            text = pytesseract.image_to_string(img)
        elif file.type == "application/pdf":
            images = convert_from_bytes(file.read())
            text = "\n".join(pytesseract.image_to_string(img.convert("RGB")) for img in images)
        return text.strip() or "No text extracted"
    except Exception as e:
        raise Exception(f"Processing failed: {e}")