
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from pypdf import PdfReader
import requests
import io
import re

app = FastAPI(title="Agentic RAG API")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Store extracted document chunks in memory.
documents = []


class ChatRequest(BaseModel):
    message: str


class RAGRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "Agentic RAG backend is running",
        "endpoints": ["/chat", "/upload", "/ask-rag"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/documents")
def list_documents():
    filenames = list(dict.fromkeys(
        doc["filename"] for doc in documents
    ))

    return {
        "uploaded_documents": filenames,
        "total_chunks": len(documents)
    }


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
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 700
                }
            },
            timeout=180
        )

        response.raise_for_status()

        return {
            "answer": response.json()["response"].strip()
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {exc}"
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is missing."
        )

    if extension not in [".txt", ".pdf"]:
        raise HTTPException(
            status_code=400,
            detail="Only TXT and PDF files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        if extension == ".txt":
            text = content.decode("utf-8-sig")

        else:
            reader = PdfReader(io.BytesIO(content))

            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF is password-protected."
                    )

            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read document: {exc}"
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text found. "
                "If this is a scanned PDF, OCR is required."
            )
        )

    # Save the uploaded file.
    save_path = UPLOAD_DIR / filename
    save_path.write_bytes(content)

    # Split the extracted text into overlapping chunks.
    chunk_size = 800
    overlap = 100
    step = chunk_size - overlap

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), step)
        if text[i:i + chunk_size].strip()
    ]

    # Replace previously uploaded chunks for the same filename.
    documents[:] = [
        doc for doc in documents
        if doc["filename"] != filename
    ]

    for index, chunk in enumerate(chunks):
        documents.append({
            "filename": filename,
            "chunk_id": index,
            "text": chunk
        })

    return {
        "message": "Document uploaded and processed successfully",
        "filename": filename,
        "pages": (
            len(reader.pages) if extension == ".pdf" else None
        ),
        "chunks_created": len(chunks),
        "characters_extracted": len(text)
    }


def tokenize(text):
    """Convert text to lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def is_summary_request(question):
    """Detect requests requiring a document overview."""
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


@app.post("/ask-rag")
def ask_rag(request: RAGRequest):
    if not documents:
        raise HTTPException(
            status_code=400,
            detail="Upload a TXT or PDF document before asking questions."
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    summary_request = is_summary_request(question)

    if summary_request:
        # Summarize the most recently uploaded document.
        latest_filename = documents[-1]["filename"]

        selected = [
            doc for doc in documents
            if doc["filename"] == latest_filename
        ]

    else:
        # Rank chunks by keyword overlap.
        question_words = {
            word for word in tokenize(question)
            if len(word) > 2
        }

        scored_chunks = []

        for doc in documents:
            chunk_words = set(tokenize(doc["text"]))

            score = sum(
                1 for word in question_words
                if word in chunk_words
            )

            if score > 0:
                scored_chunks.append((score, doc))

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True
        )

        selected = [
            doc for _, doc in scored_chunks[:6]
        ]

        if not selected:
            selected = documents[:2]

    # Build a context with a maximum size to limit model input.
    max_context_chars = 12000
    context_parts = []
    total_chars = 0

    for doc in selected:
        part = (
            f"Source: {doc['filename']}\n"
            f"Content:\n{doc['text']}"
        )

        remaining = max_context_chars - total_chars

        if remaining <= 0:
            break

        context_parts.append(part[:remaining])
        total_chars += min(len(part), remaining)

    context = "\n\n".join(context_parts)

    if summary_request:
        instruction = """
You are summarizing a document for a student.

Read the provided document context carefully and summarize
the information actually present in it.

Include:
1. The main topic and purpose of the document.
2. Important concepts, definitions, and terminology.
3. An explanation of important functions or algorithms.
4. A step-by-step explanation of code, if the document
   contains a program.
5. Important outputs, results, or conclusions, if provided.

For programming documents, explain the code in simple
language. If MPI_Reduce or MPI_Allreduce appears in the
provided context, explain their purpose and differences.

Do not say that more context is required if the supplied
context contains enough information to provide a summary.
Do not invent details that are not in the context.
If the context covers only part of a document, mention this.
"""

    else:
        instruction = """
Answer the user's question using the supplied document context.
Explain technical concepts and code in simple language.
If the answer cannot be found in the supplied context,
state that the available document text does not provide
enough information. Never invent document facts.
"""

    prompt = f"""
{instruction}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Write a direct, useful answer based on the document.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 700
                }
            },
            timeout=180
        )

        response.raise_for_status()
        result = response.json()

        return {
            "question": question,
            "answer": result["response"].strip(),
            "sources": list(dict.fromkeys(
                doc["filename"] for doc in selected
            )),
            "context_chunks_used": len(context_parts),
            "context_characters": len(context)
        }

    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {exc}"
        )
