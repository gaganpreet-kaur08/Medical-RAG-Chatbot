import chromadb
from rag.embedder import embed

client = chromadb.Client()
collection = client.get_or_create_collection(name="medical_knowledge")


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


def retrieve(query: str):
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=2
    )
    return results["documents"][0]


# 🔥 Automatically initialize knowledge when file loads
add_medical_knowledge()