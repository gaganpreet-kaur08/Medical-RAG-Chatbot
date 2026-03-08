import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file when present
load_dotenv()

# Try environment variable first
api_key = os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# If not set, and if running under Streamlit, try Streamlit secrets
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not set. Set the GROQ_API_KEY environment variable, add it to a .env file, or set Streamlit secret 'GROQ_API_KEY'.")

client = Groq(api_key=api_key)


def ask_groq(prompt: str):
    """
    Send prompt to Groq LLM and return response text.
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# Standalone test
if __name__ == "__main__":
    print("Testing Groq connection...")
    print(ask_groq("Explain what hemoglobin does in the body."))