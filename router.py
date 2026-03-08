from rag.retriever import retrieve
from llm.groq_client import ask_groq
from llm.ollama_client import ask_ollama


def generate_medical_answer(query: str, lab_context=None):
    """
    Full Multimodal RAG Pipeline:
    - Retrieves medical knowledge
    - Adds lab report understanding (if provided)
    - Routes to Groq or Ollama
    """

    print("🔍 Received query:", query)

    # Retrieve base medical knowledge
    retrieved_context = retrieve(query)
    print("📚 Retrieved context:", retrieved_context)

    combined_context = []

    if retrieved_context:
        combined_context.extend(retrieved_context)

    # Add lab-derived knowledge (multimodal input)
    if lab_context:
        print("📄 Adding lab report context...")
        combined_context.extend(lab_context)

    context_text = "\n".join(combined_context)

    print("🧠 Final combined context:", combined_context)

    prompt = f"""
You are a medical education assistant (not a doctor).

Use ONLY the information provided below to explain the findings.

Context:
{context_text}

Question:
{query}
"""

    # Hybrid routing
    if len(context_text) > 400:
        print("🧠 Using Ollama (local reasoning)...")
        return ask_ollama(prompt)
    else:
        print("⚡ Using Groq (fast inference)...")
        return ask_groq(prompt)


# Standalone test
if __name__ == "__main__":
    answer = generate_medical_answer("What does low hemoglobin indicate?")
    print("\nFinal Answer:\n", answer)