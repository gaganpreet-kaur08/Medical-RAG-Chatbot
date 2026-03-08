from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text: str):
    return model.encode(text)

if __name__ == "__main__":
    sample = "Hemoglobin is lower than normal."
    vector = embed(sample)

    print("Text:", sample)
    print("Vector length:", len(vector))
    print("First 5 values:", vector[:5])