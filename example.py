"""
Example usage of the Multimodal RAG Pipeline.

This script demonstrates how to:
1. Initialize the RAG pipeline
2. Load documents from PDF and JSON sources
3. Query the system
4. Save/load the vector index
"""

from rag_pipeline import MultimodalRAGPipeline


def main():
    # Initialize the pipeline
    print("Initializing Multimodal RAG Pipeline...")
    rag = MultimodalRAGPipeline(chunk_size=500, chunk_overlap=100)

    # Example 1: Load a PDF
    print("\n" + "="*70)
    print("LOADING PDF DOCUMENT")
    print("="*70)
    pdf_path = "multimodal_sample.pdf"
    rag.load_pdf(pdf_path, split_text=True)

    # Example 2: Load a JSON file
    print("\n" + "="*70)
    print("LOADING JSON DOCUMENT")
    print("="*70)
    json_path = "sample_data.json"
    try:
        rag.load_json(json_path, split_text=True)
    except FileNotFoundError:
        print(f"Note: {json_path} not found. Skipping JSON loading.")

    # Example 3: Query the system
    print("\n" + "="*70)
    print("QUERYING THE RAG SYSTEM")
    print("="*70)

    queries = [
        "What does the chart show about revenue trends?",
        "What are the colors used in the bar chart?",
        "Summarize the main findings from the documents",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        answer = rag.query(query, k=5, verbose=True)
        print(f"Answer: {answer}")
        print("=" * 70)

    # Example 4: Save the index
    print("\n" + "="*70)
    print("SAVING INDEX")
    print("="*70)
    rag.save_index("./vector_store_index")

    # Example 5: Load the index (in a new session)
    print("\n" + "="*70)
    print("LOADING INDEX")
    print("="*70)
    new_rag = MultimodalRAGPipeline()
    new_rag.load_index("./vector_store_index")

    # Query with loaded index
    print("\nQuerying with loaded index...")
    answer = new_rag.query("What was mentioned about Q3?", k=3, verbose=True)
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
