from typing import List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from embedder import CLIPEmbedder
from vector_store import MultimodalVectorStore


class MultimodalRetriever:
    """Retriever for multimodal documents."""

    def __init__(
        self,
        embedder: CLIPEmbedder,
        vector_store: MultimodalVectorStore
    ):
        """
        Initialize retriever.

        Args:
            embedder: CLIP embedder instance
            vector_store: Vector store instance
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.similarity_search(query_embedding, k=k)

        return results

    def create_multimodal_message(
        self,
        query: str,
        retrieved_docs: List[Document]
    ) -> HumanMessage:
        """
        Create a multimodal message with text and images for LLM.

        Args:
            query: User query
            retrieved_docs: Retrieved documents

        Returns:
            HumanMessage with text and image content
        """
        content = []

        # Add the query
        content.append({
            "type": "text",
            "text": f"Question: {query}\n\nContext:\n"
        })

        # Separate text and image documents
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'unknown')}, "
                f"Page/Index: {doc.metadata.get('page', doc.metadata.get('index', '?'))}]: "
                f"{doc.page_content}"
                for doc in text_docs
            ])
            content.append({
                "type": "text",
                "text": f"Text excerpts:\n{text_context}\n"
            })

        # Add images
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id:
                image_data = self.vector_store.get_image_data(image_id)
                if image_data:
                    content.append({
                        "type": "text",
                        "text": f"\n[Image from {doc.metadata.get('source', 'unknown')}, "
                               f"Page/Index: {doc.metadata.get('page', doc.metadata.get('index', '?'))}]:\n"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data}"
                        }
                    })

        # Add instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease answer the question based on the provided text and images."
        })

        return HumanMessage(content=content)
