import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask_ollama(prompt: str):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


if __name__ == "__main__":
    print(ask_ollama("Explain anemia in simple terms."))