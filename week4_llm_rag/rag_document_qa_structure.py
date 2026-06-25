# RAG Document Q&A Structure
# Based on Week 4 resource: Q&A with PDF/text documents using vector databases.
# This is a simplified practice version of the pipeline.

import chromadb

# Step 1: Example document text
document_text = """
Large language models are trained using next-token prediction.
Vector databases store embeddings for semantic search.
Retrieval Augmented Generation retrieves relevant chunks before answering.
Chainlit can be used to build a chat interface for LLM applications.
LangChain helps connect prompts, models, retrievers, and tools.
"""

# Step 2: Split text into chunks
chunks = [
    chunk.strip()
    for chunk in document_text.split(".")
    if chunk.strip()
]

# Step 3: Store chunks in ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="rag_practice_collection")

for index, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        metadatas=[{"source": "practice_document", "chunk": index}],
        ids=[f"chunk_{index}"]
    )

# Step 4: Ask a query
query = "What does RAG do?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

# Step 5: Retrieved context
print("Question:", query)
print("\nRetrieved relevant chunks:")
for doc in results["documents"][0]:
    print("-", doc)

print("\nThese retrieved chunks can be passed to an LLM with the question to generate an answer.")