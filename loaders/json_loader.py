import json
from PIL import Image
import base64
import io
from typing import List, Dict, Tuple
from langchain_core.documents import Document


class JSONLoader:
    """Loads and processes JSON documents with text and image URLs/paths."""

    def __init__(self, json_path: str):
        """
        Initialize JSON loader.

        Args:
            json_path: Path to the JSON file
        """
        self.json_path = json_path
        self.image_data_store = {}

    def load(self) -> Tuple[List[Document], Dict[str, str]]:
        """
        Load JSON and extract text and images.

        Expected JSON structure:
        [
            {
                "text": "Some text content",
                "metadata": {"key": "value"},
                "images": ["/path/to/image.png"] or ["data:image/png;base64,..."]
            },
            ...
        ]

        Returns:
            Tuple of (documents list, image data store dictionary)
        """
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents = []

        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]

        for idx, item in enumerate(data):
            # Process text
            text = item.get("text", "")
            metadata = item.get("metadata", {})

            if text.strip():
                text_doc = Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "index": idx,
                        "type": "text",
                        "source": self.json_path
                    }
                )
                documents.append(text_doc)

            # Process images
            images = item.get("images", [])
            for img_idx, img_data in enumerate(images):
                try:
                    image_id = f"item_{idx}_img_{img_idx}"

                    # Handle base64 encoded images
                    if isinstance(img_data, str) and img_data.startswith("data:image"):
                        # Extract base64 data
                        img_base64 = img_data.split(",")[1]
                        img_bytes = base64.b64decode(img_base64)
                        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    # Handle file paths
                    elif isinstance(img_data, str):
                        pil_image = Image.open(img_data).convert("RGB")
                        # Convert to base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    else:
                        continue

                    # Store image
                    if 'img_base64' not in locals():
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()

                    self.image_data_store[image_id] = img_base64

                    # Create document for image
                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={
                            **metadata,
                            "index": idx,
                            "type": "image",
                            "image_id": image_id,
                            "source": self.json_path,
                            "pil_image": pil_image
                        }
                    )
                    documents.append(image_doc)

                except Exception as e:
                    print(f"Error processing image {img_idx} in item {idx}: {e}")
                    continue

        return documents, self.image_data_store
