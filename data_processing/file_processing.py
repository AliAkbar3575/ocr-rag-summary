from PIL import Image
import base64
from dotenv import load_dotenv
import os
import io
import fitz

from utils import pdf_to_images, image_to_base64, ocr_page

# --------------- extracted texts from image --------------

def image_processing(image_path):
    print("="*60)
    print(f"📑 Processing IMAGE: {image_path}")
    print("="*60)

    pil_image = Image.open(image_path)
    base64_image = image_to_base64(pil_image)
    extracted_text = ocr_page(base64_image)

    return extracted_text, [extracted_text]


# ------------- text extracsion from PDF file ----------------

def pdf_processing(pdf_path):
    images = pdf_to_images(pdf_path)
    full_text = []

    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")

        b64 = image_to_base64(img)
        text = ocr_page(b64)

        full_text.append(f"\n--- Page {i+1} ---\n{text}")

    return "\n".join(full_text), full_text