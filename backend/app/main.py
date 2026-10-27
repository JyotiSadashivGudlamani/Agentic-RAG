
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from pypdf import PdfReader

import chromadb
import requests
import io
import re
import uuid
from datetime import datetime, timezone


# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

app = FastAPI(title="Agentic RAG API")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"

CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
MODEL_KEEP_ALIVE = "10m"

# Persistent ChromaDB database
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="agentic_rag_documents",
    metadata={"hnsw:space": "cosine"}
)


# =====================================================
# 2. REQUEST MODELS
# =====================================================

class ChatRequest(BaseModel):
    message: str


class RAGRequest(BaseModel):
    question: str


# =====================================================
# 3. TEXT CHUNKING
# =====================================================

def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> list[str]:
    """Split extracted text into overlapping chunks."""

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =====================================================
# 4. OLLAMA EMBEDDINGS
# =====================================================

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert text into vectors using Ollama."""

    if not texts:
        return []

    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBED_MODEL,
                "input": texts,
                "keep_alive": MODEL_KEEP_ALIVE
            },
            timeout=120
        )

        response.raise_for_status()

        embeddings = response.json().get("embeddings", [])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                "Ollama returned an unexpected number of embeddings."
            )

        return embeddings

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not generate embeddings using Ollama: {exc}"
        ) from exc


# =====================================================
# 5. DOCUMENT TEXT EXTRACTION
# =====================================================

def extract_text(filename: str, content: bytes) -> str:
    """Extract text from PDF or TXT files."""

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(content))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if extension == ".txt":
        return content.decode("utf-8-sig", errors="replace")

    raise ValueError("Only PDF and TXT files are supported.")


# =====================================================
# 6. SUMMARY REQUEST DETECTION
# =====================================================

def is_summary_request(question: str) -> bool:
    q = question.lower()

    phrases = [
        "summarize",
        "summary",
        "summarise",
        "overview",
        "explain the document",
        "explain the pdf",
        "summarize the pdf",
        "summarise the pdf",
        "what is in this pdf",
        "what is in the pdf",
        "give me the summary",
        "document summary"
    ]

    return any(phrase in q for phrase in phrases)


# =====================================================
# 7. BASIC ENDPOINTS
# =====================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to the Agentic RAG API",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBED_MODEL,
        "vector_database": "ChromaDB",
        "stored_chunks": collection.count()
    }


@app.get("/documents")
def list_documents():
    """List uploaded documents indexed in ChromaDB."""

    result = collection.get(include=["metadatas"])

    filenames = sorted({
        metadata["filename"]
        for metadata in (result.get("metadatas") or [])
        if metadata and "filename" in metadata
    })

    # Include original files that exist in uploads, even if
    # a file is not currently indexed in ChromaDB.
    for path in UPLOAD_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}:
            filenames.append(path.name)

    filenames = sorted(set(filenames))

    return {
        "documents": filenames,
        "stored_chunks": collection.count()
    }


# =====================================================
# 8. VIEW AN UPLOADED DOCUMENT
# =====================================================

@app.get("/documents/file/{filename}")
def view_document(filename: str):
    """Open an uploaded PDF or TXT file in the browser."""

    if not filename or Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_path = UPLOAD_DIR / filename

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Uploaded file not found."
        )

    extension = file_path.suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files can be viewed."
        )

    media_type = (
        "application/pdf"
        if extension == ".pdf"
        else "text/plain; charset=utf-8"
    )

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        content_disposition_type="inline"
    )


# =====================================================
# 9. DELETE AN UPLOADED DOCUMENT
# =====================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """
    Delete a document's original file and its ChromaDB chunks.
    """

    if not filename or Path(filename).name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_path = UPLOAD_DIR / filename

    # Find all indexed chunks belonging to this filename.
    existing = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )

    ids = existing.get("ids") or []

    file_exists = file_path.is_file()

    if not file_exists and not ids:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    # Delete vectors and metadata from ChromaDB.
    if ids:
        collection.delete(ids=ids)

    # Delete the original file if it exists.
    if file_exists:
        file_path.unlink()

    return {
        "message": "Document deleted successfully.",
        "filename": filename,
        "deleted_chunks": len(ids)
    }


# =====================================================
# 10. NORMAL OLLAMA CHAT
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": CHAT_MODEL,
                "prompt": message,
                "stream": False,
                "keep_alive": MODEL_KEEP_ALIVE,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 250,
                    "num_ctx": 4096
                }
            },
            timeout=180
        )

        response.raise_for_status()

        return {
            "answer": response.json().get("response", "").strip()
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Could not connect to Ollama: {exc}"
        ) from exc


# =====================================================
# 11. PDF AND TXT UPLOAD
# =====================================================

@app.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Extract, chunk, embed and store a document."""

    filename = (
        (file.filename or "")
        .replace("\\", "/")
        .split("/")[-1]
    )

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if Path(filename).suffix.lower() not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported."
        )

    try:
        content = file.file.read()

        text = extract_text(filename, content)

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "No readable text was found. "
                    "If this is a scanned PDF, OCR may be required."
                )
            )

        chunks = split_text(text)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created."
            )

        # Generate embeddings in batches.
        embeddings = []

        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            embeddings.extend(get_embeddings(batch))

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                "Not all document chunks were embedded."
            )

        # Replace any existing indexed document with the same filename.
        existing = collection.get(
            where={"filename": filename},
            include=["metadatas"]
        )

        upload_time = datetime.now(timezone.utc).isoformat()

        ids = [str(uuid.uuid4()) for _ in chunks]

        metadatas = [
            {
                "filename": filename,
                "chunk_index": index,
                "uploaded_at": upload_time
            }
            for index in range(len(chunks))
        ]

        # Store the new chunks.
        # Use fresh UUIDs so ChromaDB IDs never conflict.
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Remove any old chunks for this filename that were not
        # included in the new upload.
        old_ids = existing.get("ids") or []
        if old_ids:
            collection.delete(ids=old_ids)

        # Save the original uploaded file.
        save_path = UPLOAD_DIR / filename
        save_path.write_bytes(content)

        return {
            "message": "Document uploaded and indexed successfully.",
            "filename": filename,
            "pages": (
                len(PdfReader(io.BytesIO(content)).pages)
                if Path(filename).suffix.lower() == ".pdf"
                else None
            ),
            "characters_extracted": len(text),
            "chunks_created": len(chunks),
            "vectors_stored": len(embeddings),
            "vector_database": "ChromaDB"
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {exc}"
        ) from exc

    finally:
        file.file.close()


# =====================================================
# 12. SEMANTIC DOCUMENT RETRIEVAL AND RAG
# =====================================================

@app.post("/ask-rag")
def ask_rag(request: RAGRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    total_chunks = collection.count()

    if total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF or TXT file first."
        )

    try:
        summary_request = is_summary_request(question)

        if summary_request:
            # Summarize the most recently uploaded indexed document.
            all_data = collection.get(
                include=["documents", "metadatas"]
            )

            metadata_list = all_data.get("metadatas") or []
            document_list = all_data.get("documents") or []

            if not metadata_list or not document_list:
                raise HTTPException(
                    status_code=404,
                    detail="No indexed document content was found."
                )

            latest_filename = max(
                metadata_list,
                key=lambda metadata: metadata.get("uploaded_at", "")
            ).get("filename")

            selected_pairs = [
                (document, metadata)
                for document, metadata in zip(
                    document_list, metadata_list
                )
                if metadata.get("filename") == latest_filename
            ]

            selected_pairs.sort(
                key=lambda pair: pair[1].get("chunk_index", 0)
            )

            selected_documents = [
                document for document, _ in selected_pairs
            ]

            selected_metadata = [
                metadata for _, metadata in selected_pairs
            ]

        else:
            # Embed the question and retrieve the three closest chunks.
            question_embedding = get_embeddings([question])[0]

            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=min(3, total_chunks),
                include=[
                    "documents",
                    "metadatas",
                    "distances"
                ]
            )

            selected_documents = results["documents"][0] or []
            selected_metadata = results["metadatas"][0] or []

        if not selected_documents:
            raise HTTPException(
                status_code=404,
                detail="No relevant document content was found."
            )

        # Limit context to reduce prompt processing time.
        max_context_chars = 6000
        context_parts = []
        total_chars = 0

        for document, metadata in zip(
            selected_documents, selected_metadata
        ):
            part = (
                f"Source: {metadata.get('filename', 'Unknown file')}\n"
                f"Content:\n{document}"
            )

            remaining = max_context_chars - total_chars

            if remaining <= 0:
                break

            context_parts.append(part[:remaining])
            total_chars += min(len(part), remaining)

        context = "\n\n".join(context_parts)

        if summary_request:
            instruction = """
You are a concise document assistant.

Summarize the supplied document context in simple language.
Include the main topic and the most important concepts.
Explain technical terms only when relevant.
For programming documents, explain important code or algorithms
when they appear in the supplied context.
Do not invent facts. If only part of the document is available,
mention this limitation.
Keep the summary concise, using short paragraphs or bullet points.
"""
            max_tokens = 350

        else:
            instruction = """
Answer the user's question using the supplied document context.
Be accurate, concise, and easy to understand.
Use a short paragraph or a few bullet points.
Do not invent facts or make unsupported assumptions.
If the answer is not present in the context, say so clearly.
"""
            max_tokens = 250

        prompt = f"""
{instruction}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a direct answer without unnecessary repetition.
"""

        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": MODEL_KEEP_ALIVE,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_tokens,
                    "num_ctx": 4096
                }
            },
            timeout=180
        )

        response.raise_for_status()

        answer = response.json().get("response", "").strip()

        return {
            "question": question,
            "answer": answer,
            "sources": sorted({
                metadata.get("filename", "Unknown file")
                for metadata in selected_metadata
            }),
            "context_chunks_used": len(context_parts),
            "context_characters": len(context),
            "search_type": (
                "document summary"
                if summary_request
                else "semantic similarity"
            ),
            "vector_database": "ChromaDB"
        }

    except HTTPException:
        raise

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama request failed: {exc}"
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {exc}"
        ) from exc
