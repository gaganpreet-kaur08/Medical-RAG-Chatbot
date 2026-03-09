from llm.groq_client import ask_groq

def generate_text(prompt: str):
    """
    Generate text using Groq LLM.
    """
    return ask_groq(prompt)