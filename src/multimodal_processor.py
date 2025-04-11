import pytesseract
from PIL import Image
import pdf2image
import os
from loguru import logger

def process_multimodal_input(file_path):
    """
    Process a PDF file to extract text using OCR.
    Args:
        file_path (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """
    logger.debug(f"Starting multimodal processing for file: {file_path}")
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # Convert PDF to images with high DPI
        logger.debug("Converting PDF to images")
        images = pdf2image.convert_from_path(file_path, dpi=300, poppler_path=None)
        if not images:
            raise ValueError("No images extracted from PDF")
        extracted_text = ""
        for i, image in enumerate(images):
            logger.debug(f"Processing page {i+1}")
            text = pytesseract.image_to_string(image, lang="eng", config="--psm 6")
            extracted_text += f"\nPage {i+1}:\n{text}\n"
            logger.info(f"Extracted {len(text)} characters from page {i+1}")
        logger.info(f"Completed processing {file_path}")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error in process_multimodal_input: {str(e)}")
        return f"Error processing file: {str(e)}. Please ensure the PDF is valid and dependencies are installed."

def enhance_ocr_text(raw_text):
    """
    Enhance raw OCR text by removing noise and improving readability.
    Args:
        raw_text (str): Raw text from OCR.
    Returns:
        str: Cleaned and enhanced text.
    """
    logger.debug("Starting text enhancement")
    try:
        if not raw_text:
            raise ValueError("No text provided for enhancement")
        cleaned_text = " ".join(raw_text.split())
        cleaned_text = cleaned_text.replace("\n\n\n", "\n").replace("\t", " ")
        cleaned_text = cleaned_text.replace("contrct", "contract").replace("penatly", "penalty")
        logger.info("Text enhancement completed")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error in enhance_ocr_text: {str(e)}")
        return raw_text

if __name__ == "__main__":
    if os.path.exists("uploaded_document.pdf"):
        logger.info("Testing multimodal_processor with uploaded_document.pdf")
        text = process_multimodal_input("uploaded_document.pdf")
        enhanced_text = enhance_ocr_text(text)
        with open("processed_output.txt", "w", encoding="utf-8") as f:
            f.write(enhanced_text)
        logger.info("Test output saved to processed_output.txt")
        print("Processed text sample:", enhanced_text[:200])
    else:
        logger.warning("No uploaded_document.pdf found for testing")