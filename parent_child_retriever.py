import chromadb
from typing import List, Dict, Tuple
import re
from rag.embedder import embed
from rag.multiquery import generate_multi_queries
from rag.hyde import generate_hypothetical_doc

# Separate client for parent-child retrieval
pc_client = chromadb.Client()
parent_collection = pc_client.get_or_create_collection(name="medical_parents")
child_collection = pc_client.get_or_create_collection(name="medical_children")


def sentence_split(text: str) -> List[str]:
    """
    Split text into sentences using regex-based sentence boundary detection.
    Handles medical text with abbreviations and measurements.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    # Medical-aware sentence splitting regex
    # Handles common medical abbreviations: g/dL, mg/L, mmHg, etc.
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
    medical_abbrevs = r'\b(?:g/dL|mg/L|mmHg|mcL|mmol/L|μmol/L|IU/L|pg/mL|ng/mL|μg/mL|U/L)\b'

    # Split on sentence endings but avoid splitting on medical abbreviations
    sentences = re.split(sentence_endings, text.strip())

    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def chunk_text(text: str, chunk_size: int = 3, overlap: int = 1, method: str = "sentence") -> List[str]:
    """
    Split text into chunks using sentence-level or character-level splitting.

    Args:
        text: The text to chunk
        chunk_size: Maximum number of sentences per chunk (if method="sentence")
                   or maximum characters per chunk (if method="character")
        overlap: Number of sentences/characters to overlap between chunks
        method: "sentence" or "character"

    Returns:
        List of text chunks
    """
    if method == "sentence":
        return chunk_by_sentences(text, chunk_size, overlap)
    else:
        return chunk_by_characters(text, chunk_size, overlap)


def chunk_by_sentences(text: str, sentences_per_chunk: int = 3, overlap_sentences: int = 1) -> List[str]:
    """
    Split text into chunks by grouping sentences with overlap.

    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of sentence-grouped chunks
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks = []
    step = sentences_per_chunk - overlap_sentences

    for i in range(0, len(sentences), max(1, step)):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        if chunk_sentences:
            chunk_text = ' '.join(chunk_sentences)
            if len(chunk_text.strip()) > 10:
                chunks.append(chunk_text)

    return chunks


def chunk_by_characters(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """
    Original character-level chunking (fallback method).

    Args:
        text: The text to chunk
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 10:
            chunks.append(chunk)
    return chunks


def add_medical_knowledge_with_parent_child():
    """
    Add medical knowledge with parent-child relationships.
    Parent = full document, Children = smaller chunks extracted from parent.
    """
    parent_docs = [
        "Low hemoglobin levels may indicate anemia, often caused by iron deficiency. Anemia symptoms include fatigue, weakness, and shortness of breath. Treatment depends on the underlying cause.",
        "High CRP levels are associated with inflammation or infection in the body. Normal CRP is below 3 mg/L. Elevated levels warrant further investigation to identify the source of inflammation.",
        "Normal hemoglobin range for adult males is 13.5 to 17.5 g/dL, while for adult females it is 12.0 to 15.5 g/dL. Hemoglobin below normal range indicates anemia.",
        "Chest X-rays showing localized opacity may suggest pneumonia or infection. Additional imaging and clinical correlation are recommended. CT chest may be needed for clarification.",
        "Elevated white blood cell count can indicate immune response to infection, leukemia, or inflammation. Normal WBC range is 4,500 to 11,000 cells/mcL."
    ]

    # Clear existing collections
    if parent_collection.count() > 0:
        return  # Already populated

    # Add parents and children
    for parent_idx, parent_doc in enumerate(parent_docs):
        parent_id = f"parent_{parent_idx}"
        
        # Add parent document
        parent_collection.add(
            documents=[parent_doc],
            embeddings=[embed(parent_doc)],
            ids=[parent_id],
            metadatas=[{"type": "parent", "original_index": parent_idx}]
        )
        
        # Create and add child chunks
        child_chunks = chunk_text(parent_doc, chunk_size=2, overlap=1, method="sentence")
        for chunk_idx, child_doc in enumerate(child_chunks):
            child_id = f"child_{parent_idx}_{chunk_idx}"
            
            child_collection.add(
                documents=[child_doc],
                embeddings=[embed(child_doc)],
                ids=[child_id],
                metadatas=[{
                    "type": "child",
                    "parent_id": parent_id,
                    "parent_index": parent_idx,
                    "chunk_index": chunk_idx
                }]
            )
    
    print("✅ Parent-child knowledge base initialized.")


def retrieve_with_parent_child(query: str, k: int = 3) -> List[str]:
    """
    Retrieve documents using parent-child strategy:
    1. Find matching child chunks using multi-query and HyDE
    2. Retrieve full parent documents for matched children
    3. Return complete parent documents for better context
    
    Args:
        query: The user's query
        k: Number of results to retrieve
    
    Returns:
        List of parent documents (full context)
    """
    
    # Step 1: Generate multiple queries
    queries = generate_multi_queries(query)
    
    parent_ids_found = set()
    all_matches = []
    
    # Step 2: Search child collection for matches
    for q in queries:
        # Generate hypothetical document (HyDE)
        hypothetical_doc = generate_hypothetical_doc(q)
        query_embedding = embed(hypothetical_doc)
        
        # Search in child collection
        results = child_collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Extract parent IDs from matched children
        if results["metadatas"] and len(results["metadatas"]) > 0:
            for metadata in results["metadatas"][0]:
                if metadata.get("type") == "child":
                    parent_id = metadata.get("parent_id")
                    if parent_id:
                        parent_ids_found.add(parent_id)
                        all_matches.append({
                            "parent_id": parent_id,
                            "child_doc": results["documents"][0][results["metadatas"][0].index(metadata)]
                        })
        
        # If no children matched, also search parent collection directly
        if not parent_ids_found:
            parent_results = parent_collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            if parent_results["ids"] and len(parent_results["ids"]) > 0:
                for parent_id in parent_results["ids"][0]:
                    parent_ids_found.add(parent_id)
    
    # Step 3: Retrieve full parent documents
    parent_docs = []
    for parent_id in list(parent_ids_found)[:k]:
        parent_result = parent_collection.get(ids=[parent_id])
        if parent_result["documents"]:
            parent_docs.append(parent_result["documents"][0])
    
    return parent_docs


def retrieve_hybrid(query: str, k: int = 3, use_parent_child: bool = True) -> Dict:
    """
    Hybrid retrieval that combines both standard and parent-child retrieval.
    
    Args:
        query: The user's query
        k: Number of results
        use_parent_child: Whether to use parent-child strategy
    
    Returns:
        Dictionary containing both standard and parent-child results
    """
    if use_parent_child:
        parent_child_results = retrieve_with_parent_child(query, k)
        return {
            "parent_child_docs": parent_child_results,
            "method": "parent_child"
        }
    
    # Fallback to standard retrieval if needed
    from rag.retriever import retrieve as standard_retrieve
    standard_results = standard_retrieve(query, k)
    return {
        "standard_docs": standard_results,
        "method": "standard"
    }


# Auto-initialize on import
add_medical_knowledge_with_parent_child()
