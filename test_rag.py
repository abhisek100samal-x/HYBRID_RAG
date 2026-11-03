from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

print("1. Creating test document...")

documents = [
    Document(
        page_content="The capital of France is Paris.",
        metadata={"source": "test_document.txt", "page": 1},
    )
]

print("2. Connecting to Ollama embeddings...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

print("3. Indexing document in ChromaDB...")
store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="rag_pipeline_test",
)

print("4. Retrieving relevant passages...")
question = "What is the capital of France?"
results = store.similarity_search(question, k=1)

for doc in results:
    print("Retrieved:", doc.page_content)

print("5. Generating answer with Llama...")
context = "\n".join(doc.page_content for doc in results)

llm = ChatOllama(model="llama3.2:latest", temperature=0)
response = llm.invoke(
    f"Answer using only this context:\n{context}\nQuestion: {question}"
)

print("ANSWER:", response.content)
print("SUCCESS: End-to-end RAG pipeline works!")
