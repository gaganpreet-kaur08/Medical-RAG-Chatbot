import chromadb
from rag.embedder import embed
from rag.multiquery import generate_multi_queries
from rag.hyde import generate_hypothetical_doc

client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_knowledge")

# Retrieval mode: "standard" or "parent_child"
RETRIEVAL_MODE = "parent_child"


def add_medical_knowledge():
    docs = [
        "Low hemoglobin levels may indicate anemia, often caused by iron deficiency.",
        "High CRP levels are associated with inflammation or infection in the body.",
        "Normal hemoglobin range for adult males is 13.5 to 17.5 g/dL.",
        "Chest X-rays showing localized opacity may suggest pneumonia or infection.",
        "Elevated white blood cell count can indicate immune response to infection."
    ]

    # Only add if empty (prevents duplicates)
    if collection.count() == 0:
        for i, doc in enumerate(docs):
            collection.add(
                documents=[doc],
                embeddings=[embed(doc)],
                ids=[f"doc_{i}"]
            )
        print("✅ Knowledge base initialized.")


def retrieve_standard(query: str, k: int = 3):
    """
    Standard retrieval using:
    1. MultiQuery generation
    2. HyDE hypothetical document embeddings
    """

    # Step 1: Generate multiple queries
    queries = generate_multi_queries(query)

    all_docs = []

    # Step 2: Retrieve documents for each query
    for q in queries:

        # Step 2A: Generate hypothetical document (HyDE)
        hypothetical_doc = generate_hypothetical_doc(q)

        # Step 2B: Convert to embedding
        query_embedding = embed(hypothetical_doc)

        # Step 2C: Search vector database
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        docs = results["documents"][0]

        all_docs.extend(docs)

    # Step 3: Remove duplicates
    unique_docs = list(set(all_docs))

    return unique_docs


def retrieve(query: str, k: int = 3, mode: str = None):
    """
    Retrieve relevant documents using the configured retrieval mode.
    
    Args:
        query: The search query
        k: Number of results to return
        mode: Override retrieval mode ("standard" or "parent_child"). 
              If None, uses RETRIEVAL_MODE setting.
    
    Returns:
        List of relevant documents
    """
    retrieval_mode = mode if mode else RETRIEVAL_MODE
    
    if retrieval_mode == "parent_child":
        from rag.parent_child_retriever import retrieve_with_parent_child
        return retrieve_with_parent_child(query, k)
    else:
        return retrieve_standard(query, k)


# 🔥 Automatically initialize knowledge when file loads
add_medical_knowledge()