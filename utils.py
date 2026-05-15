# from openai import OpenAI
from PIL import Image
import base64
from dotenv import load_dotenv
import os
import io
import fitz  # PyMuPDF

from groq import Groq
from dotenv import load_dotenv
load_dotenv()


# ---------- make a list of images from PDF file ------------------

def pdf_to_images(pdf_path, zoom=2):
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)

    return images


# ----------- convert PIL image to base64 ------------

def image_to_base64(pil_image):
    # Fix all weird modes: P, RGBA, L, etc.
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=95)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ----------- text extraction from image --------------

def ocr_page(base64_image):

    client = Groq()
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a strict OCR transcription engine.

                                Task:
                                Transcribe the handwritten text exactly as it appears in the image.

                                Rules:
                                - Copy text character-by-character.
                                - Preserve original spelling mistakes.
                                - Preserve capitalization.
                                - Preserve punctuation.
                                - Preserve line breaks.
                                - Do NOT correct grammar or spelling.
                                - Do NOT summarize.
                                - Do NOT explain anything.
                                - Do NOT add missing words.
                                - Do NOT infer unclear text.
                                - Output ONLY the transcription text.
                                - Do not include comments or extra sentences.
                                - Reconstruct tables using markdown format:
                                    Example:
                                    | Column1 | Column2 |
                                    |--------|--------|
                                    | value1 | value2 |
                                - For images, graphs, figues: Only extract any visible text inside or around them.
                                - Finally, Output ONLY the extracted content. No explanations.
                                
                                """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=2048
    )

    return completion.choices[0].message.content

def save_as_text(variale_name, destination):
    with open(destination, "w", encoding="utf-8") as f:
        f.write(variale_name)
    print(f"file saved to - {destination}")