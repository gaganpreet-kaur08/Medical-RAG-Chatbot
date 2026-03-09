

from rag.parent_child_retriever import sentence_split, chunk_by_sentences, chunk_text

# Test text with medical content
test_text = "Low hemoglobin levels may indicate anemia, often caused by iron deficiency. Anemia symptoms include fatigue, weakness, and shortness of breath. Treatment depends on the underlying cause."

print("🔍 Testing Sentence-Level Text Splitter")
print("=" * 60)

print("\n📝 Original Text:")
print(test_text)
print()

print("✂️ Sentence Splitting:")
sentences = sentence_split(test_text)
for i, sentence in enumerate(sentences, 1):
    print(f"  {i}. {sentence}")

print(f"\n📊 Total sentences: {len(sentences)}")

print("\n📦 Sentence-Level Chunking (2 sentences per chunk, 1 overlap):")
chunks = chunk_by_sentences(test_text, sentences_per_chunk=2, overlap_sentences=1)
for i, chunk in enumerate(chunks, 1):
    print(f"  Chunk {i}: {chunk}")
    print(f"    Length: {len(chunk)} characters")

print(f"\n📊 Total chunks: {len(chunks)}")

print("\n🔄 Unified chunk_text() function (sentence method):")
chunks_unified = chunk_text(test_text, chunk_size=2, overlap=1, method="sentence")
for i, chunk in enumerate(chunks_unified, 1):
    print(f"  Chunk {i}: {chunk}")

print("\n✅ Sentence-level splitter test completed!")

# Test with medical abbreviations
print("\n🏥 Testing with Medical Abbreviations:")
medical_text = "Normal hemoglobin range for adult males is 13.5 to 17.5 g/dL. For adult females it is 12.0 to 15.5 g/dL. Hemoglobin below normal range indicates anemia."
print(f"Text: {medical_text}")

med_sentences = sentence_split(medical_text)
print("Sentences:")
for i, s in enumerate(med_sentences, 1):
    print(f"  {i}. {s}")

med_chunks = chunk_text(medical_text, chunk_size=2, overlap=1, method="sentence")
print("Chunks:")
for i, c in enumerate(med_chunks, 1):
    print(f"  {i}. {c}")
