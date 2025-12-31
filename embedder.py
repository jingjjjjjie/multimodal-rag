import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Union, List


class CLIPEmbedder:
    """Embedder using CLIP for both text and images."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP embedder.

        Args:
            model_name: HuggingFace model name for CLIP
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def embed_image(self, image_data: Union[str, Image.Image]) -> np.ndarray:
        """
        Embed image using CLIP.

        Args:
            image_data: Either a file path or PIL Image

        Returns:
            Normalized embedding vector
        """
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data

        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # Normalize embeddings to unit vector
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed text using CLIP.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector
        """
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            # Normalize embeddings
            features = features / features.norm(dim=-1, keepdim=True)
            return features.squeeze().numpy()

    def embed_documents(self, documents: List) -> List[np.ndarray]:
        """
        Embed a list of documents (text or images).

        Args:
            documents: List of Document objects

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for doc in documents:
            if doc.metadata.get("type") == "image":
                # Use PIL image from metadata if available
                pil_image = doc.metadata.get("pil_image")
                if pil_image:
                    embedding = self.embed_image(pil_image)
                else:
                    # Fallback to text embedding if no image available
                    embedding = self.embed_text(doc.page_content)
            else:
                embedding = self.embed_text(doc.page_content)
            embeddings.append(embedding)
        return embeddings
