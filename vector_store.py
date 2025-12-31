import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


class MultimodalVectorStore:
    """Vector store for multimodal documents using FAISS."""

    def __init__(self):
        """Initialize the vector store."""
        self.vector_store = None
        self.documents = []
        self.image_data_store = {}

    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[np.ndarray],
        image_data_store: Dict[str, str] = None
    ):
        """
        Add documents with their embeddings to the vector store.

        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
            image_data_store: Dictionary mapping image IDs to base64 data
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Store documents
        self.documents.extend(documents)

        # Store image data
        if image_data_store:
            self.image_data_store.update(image_data_store)

        # Create or update FAISS index
        embeddings_array = np.array(embeddings)
        text_embeddings = [
            (doc.page_content, emb)
            for doc, emb in zip(documents, embeddings_array)
        ]
        metadatas = [doc.metadata for doc in documents]

        if self.vector_store is None:
            # Create new vector store
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=None,
                metadatas=metadatas
            )
        else:
            # Add to existing vector store
            new_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=None,
                metadatas=metadatas
            )
            self.vector_store.merge_from(new_store)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Document]:
        """
        Search for similar documents using query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            return []

        results = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=k
        )
        return results

    def get_image_data(self, image_id: str) -> str:
        """
        Get base64 image data by image ID.

        Args:
            image_id: Image identifier

        Returns:
            Base64 encoded image string
        """
        return self.image_data_store.get(image_id)

    def save(self, path: str):
        """
        Save the vector store and metadata to disk.

        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store:
            save_path = Path(path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save FAISS index
            self.vector_store.save_local(str(save_path / "faiss_index"))

            # Save image data store and documents metadata
            metadata = {
                "image_data_store": self.image_data_store,
                "documents": self.documents
            }
            with open(save_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

    def load(self, path: str):
        """
        Load the vector store and metadata from disk.

        Args:
            path: Directory path to load the vector store from
        """
        load_path = Path(path)

        # Load FAISS index
        self.vector_store = FAISS.load_local(
            str(load_path / "faiss_index"),
            embeddings=None,
            allow_dangerous_deserialization=True
        )

        # Load image data store and documents metadata
        with open(load_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            self.image_data_store = metadata.get("image_data_store", {})
            self.documents = metadata.get("documents", [])
