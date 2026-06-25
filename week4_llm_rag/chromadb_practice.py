# ChromaDB Basic Vector Database Practice
# Based on Week 4 resource: Vector databases and embeddings.

import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="week4_vector_database_practice")

collection.add(
    documents=[
        "Large language models generate text using next-token prediction.",
        "Vector databases store embeddings and help with semantic search.",
        "RAG retrieves relevant document chunks before generating an answer.",
        "Chainlit helps create chat-based user interfaces for LLM applications."
    ],
    metadatas=[
        {"source": "llm_intro_notes"},
        {"source": "vector_database_notes"},
        {"source": "rag_notes"},
        {"source": "chainlit_notes"}
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

results = collection.query(
    query_texts=["What is used to store embeddings for semantic search?"],
    n_results=2
)

print(results)