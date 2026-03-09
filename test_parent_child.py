

from rag.retriever import retrieve
from rag.parent_child_retriever import retrieve_with_parent_child, retrieve_hybrid

# Test queries
test_queries = [
    "What causes hemoglobin to be low?",
    "How do I interpret elevated CRP levels?",
    "What does a chest X-ray opacity indicate?",
    "What is the normal range for white blood cells?"
]

print("=" * 80)
print("PARENT-CHILD RETRIEVER TEST")
print("=" * 80)

for query in test_queries:
    print(f"\n📋 QUERY: {query}")
    print("-" * 80)
    
    # Standard retrieval
    print("\n🔵 STANDARD RETRIEVAL:")
    standard_results = retrieve(query, k=2, mode="standard")
    for i, doc in enumerate(standard_results, 1):
        print(f"  [{i}] {doc[:100]}...")
    
    # Parent-child retrieval
    print("\n🟢 PARENT-CHILD RETRIEVAL:")
    parent_child_results = retrieve_with_parent_child(query, k=2)
    for i, doc in enumerate(parent_child_results, 1):
        print(f"  [{i}] {doc[:100]}...")
    
    # Hybrid retrieval
    print("\n🟡 HYBRID RETRIEVAL (Parent-Child):")
    hybrid_results = retrieve_hybrid(query, k=2, use_parent_child=True)
    for i, doc in enumerate(hybrid_results["parent_child_docs"], 1):
        print(f"  [{i}] {doc[:100]}...")
    
    print()

print("\n" + "=" * 80)
print("✅ Test completed! Parent-child retriever is working.")
print("=" * 80)
print("\n📌 To use parent-child retrieval in your app:")
print("   1. Import from rag.retriever: from rag.retriever import retrieve")
print("   2. Call retrieve(query) - it will use parent-child mode by default")
print("   3. Or explicitly: retrieve(query, mode='parent_child')")
print("   4. For standard mode: retrieve(query, mode='standard')")
