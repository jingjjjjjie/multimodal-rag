import fitz
from PIL import Image
import io
import base64
from typing import List, Dict, Tuple
from langchain_core.documents import Document


class PDFLoader:
    """Loads and processes PDF documents with text and images."""

    def __init__(self, pdf_path: str):
        """
        Initialize PDF loader.

        Args:
            pdf_path: Path to the PDF file
        """
        self.pdf_path = pdf_path
        self.image_data_store = {}

    def load(self) -> Tuple[List[Document], Dict[str, str]]:
        """
        Load PDF and extract text and images.

        Returns:
            Tuple of (documents list, image data store dictionary)
        """
        doc = fitz.open(self.pdf_path)
        documents = []

        for page_num, page in enumerate(doc):
            # Process text
            text = page.get_text()
            if text.strip():
                text_doc = Document(
                    page_content=text,
                    metadata={
                        "page": page_num,
                        "type": "text",
                        "source": self.pdf_path
                    }
                )
                documents.append(text_doc)

            # Process images
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    # Create unique identifier
                    image_id = f"page_{page_num}_img_{img_index}"

                    # Store image as base64
                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    self.image_data_store[image_id] = img_base64

                    # Create document for image
                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={
                            "page": page_num,
                            "type": "image",
                            "image_id": image_id,
                            "source": self.pdf_path,
                            "pil_image": pil_image
                        }
                    )
                    documents.append(image_doc)

                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num}: {e}")
                    continue

        doc.close()
        return documents, self.image_data_store
