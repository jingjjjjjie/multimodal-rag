from typing import List, Dict, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from loaders import PDFLoader, JSONLoader
from embedder import CLIPEmbedder
from vector_store import MultimodalVectorStore
from retriever import MultimodalRetriever
from models import initialize_qwen_multimodal_api


class MultimodalRAGPipeline:
    """Main pipeline for multimodal RAG."""

    def __init__(
        self,
        embedder: Optional[CLIPEmbedder] = None,
        llm = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize the RAG pipeline.

        Args:
            embedder: CLIP embedder instance (creates default if None)
            llm: Language model instance (creates default if None)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
        """
        self.embedder = embedder or CLIPEmbedder()
        self.llm = llm or initialize_qwen_multimodal_api()
        self.vector_store = MultimodalVectorStore()
        self.retriever = MultimodalRetriever(self.embedder, self.vector_store)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_pdf(self, pdf_path: str, split_text: bool = True):
        """
        Load and process a PDF document.

        Args:
            pdf_path: Path to PDF file
            split_text: Whether to split text into chunks
        """
        loader = PDFLoader(pdf_path)
        documents, image_data_store = loader.load()

        # Split text documents if requested
        if split_text:
            text_docs = [doc for doc in documents if doc.metadata.get("type") == "text"]
            image_docs = [doc for doc in documents if doc.metadata.get("type") == "image"]

            # Split text documents
            split_docs = self.text_splitter.split_documents(text_docs)

            # Combine split text and images
            documents = split_docs + image_docs

        # Generate embeddings
        embeddings = self.embedder.embed_documents(documents)

        # Add to vector store
        self.vector_store.add_documents(documents, embeddings, image_data_store)

        print(f"Loaded PDF: {pdf_path}")
        print(f"  - Added {len(documents)} documents ({len([d for d in documents if d.metadata.get('type') == 'text'])} text, {len([d for d in documents if d.metadata.get('type') == 'image'])} images)")

    def load_json(self, json_path: str, split_text: bool = True):
        """
        Load and process a JSON document.

        Args:
            json_path: Path to JSON file
            split_text: Whether to split text into chunks
        """
        loader = JSONLoader(json_path)
        documents, image_data_store = loader.load()

        # Split text documents if requested
        if split_text:
            text_docs = [doc for doc in documents if doc.metadata.get("type") == "text"]
            image_docs = [doc for doc in documents if doc.metadata.get("type") == "image"]

            # Split text documents
            split_docs = self.text_splitter.split_documents(text_docs)

            # Combine split text and images
            documents = split_docs + image_docs

        # Generate embeddings
        embeddings = self.embedder.embed_documents(documents)

        # Add to vector store
        self.vector_store.add_documents(documents, embeddings, image_data_store)

        print(f"Loaded JSON: {json_path}")
        print(f"  - Added {len(documents)} documents ({len([d for d in documents if d.metadata.get('type') == 'text'])} text, {len([d for d in documents if d.metadata.get('type') == 'image'])} images)")

    def query(self, query: str, k: int = 5, verbose: bool = True) -> str:
        """
        Query the RAG system.

        Args:
            query: User query
            k: Number of documents to retrieve
            verbose: Whether to print retrieval information

        Returns:
            Answer from the LLM
        """
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=k)

        if verbose:
            print(f"\nRetrieved {len(retrieved_docs)} documents:")
            for doc in retrieved_docs:
                doc_type = doc.metadata.get("type", "unknown")
                source = doc.metadata.get("source", "unknown")
                page_or_idx = doc.metadata.get("page", doc.metadata.get("index", "?"))

                if doc_type == "text":
                    preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
                    print(f"  - Text from {source} (page/index {page_or_idx}): {preview}")
                else:
                    print(f"  - Image from {source} (page/index {page_or_idx})")
            print()

        # Create multimodal message
        message = self.retriever.create_multimodal_message(query, retrieved_docs)

        # Get response from LLM
        response = self.llm.invoke([message])

        return response.content

    def save_index(self, path: str):
        """
        Save the vector store index.

        Args:
            path: Directory path to save the index
        """
        self.vector_store.save(path)
        print(f"Saved index to: {path}")

    def load_index(self, path: str):
        """
        Load a saved vector store index.

        Args:
            path: Directory path to load the index from
        """
        self.vector_store.load(path)
        print(f"Loaded index from: {path}")
