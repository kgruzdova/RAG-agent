"""
Точка входа: smoke test RAG-агента.
"""

from rag_agent import RAGAgent


if __name__ == "__main__":
    print("=" * 60)
    print("RAGAgent — smoke test")
    print("=" * 60)

    try:
        agent = RAGAgent()
    except ValueError as exc:
        print(f"[CONFIG ERROR] {exc}")
        print("Please fill in the required values in your .env file.")
        raise SystemExit(1)

    TEST_QUERY = "What is this knowledge base about?"
    print(f"\nRunning test query: '{TEST_QUERY}'")
    print("(The actual answer is not important — we just verify connectivity.)\n")

    try:
        results = agent.similarity_search(TEST_QUERY, k=1)
        if results:
            print(f"[OK] Pinecone connection successful. Retrieved {len(results)} result(s).")
            print(f"     First result preview: {results[0].page_content[:120]}...")
        else:
            print("[OK] Pinecone connection successful. Index is empty (no results returned).")
    except Exception as exc:
        print(f"[FAIL] Could not connect to Pinecone: {exc}")
        raise SystemExit(1)

    print("\nSmoke test passed.")
