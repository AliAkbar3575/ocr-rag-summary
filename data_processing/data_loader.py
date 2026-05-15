from PIL import Image
from pathlib import Path
import time

from data_processing.file_processing import image_processing, pdf_processing
from utils import save_as_text


# ----------------- file path defining ---------------

file_path = "data/pdfs/doc.pdf" # for pdf loading
file_path = "data/images/img1.jpg" # for image loading

# ------------- check file extension and relevant action ----------

print("🔃 File procesing started......")
time.sleep(3)

extension = Path(file_path).suffix.lower()

if extension in [".jpg", ".png"]:
    extracted_text, extracted_text_list = image_processing(file_path)

    save_as_text(extracted_text, "./output/text_from_image.txt")

elif extension == ".pdf":
    extracted_text, extracted_text_list = pdf_processing(file_path)

    save_as_text(extracted_text, "./output/text_from_pdf.txt")

else:
    print("❌ invalid file! Please insert .jpg/.png/.pdf file...")




