from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()

# Connect to the same ChromaDB collection you built in Step 2
client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


# Define the expected shape of incoming data for the POST endpoint
class DocumentSubmission(BaseModel):
    user_name: str  # Who this profile belongs to
    content: str    # The profile text to store

    @field_validator("user_name", "content")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Field cannot be empty or whitespace.")
        return v.strip()


@app.post("/documents")  # POST endpoint - accepts data in the request body
def add_document(submission: DocumentSubmission):
    # Split the submitted profile into chunks by paragraph
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    if not chunks:
        raise HTTPException(status_code=400, detail="No valid chunks found in content.")

    # Delete existing chunks for this user to avoid duplicate ID errors on resubmission
    existing = collection.get(where={"user_name": submission.user_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Store each chunk in ChromaDB with user metadata and chunk position
    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }


@app.get("/ask")
def ask(question: str):
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1: RETRIEVE - search ChromaDB for the 2 most relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=2,
    )

    # Combine the matching chunks into a single string
    context = "\n\n".join(results["documents"][0])

    # Step 2: AUGMENT - build a prompt that includes the retrieved context
    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""

    # Step 3: GENERATE - send the augmented prompt to the local LLM
    response = ollama.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer along with the context so users can verify the source
    return {
        "question": question,
        "answer": response.message.content,
        "context_used": results["documents"][0],
    }