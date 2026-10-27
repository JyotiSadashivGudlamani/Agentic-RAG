# Agentic RAG Web Application

A full-stack AI-powered web application that allows users to upload PDF and TXT documents, ask natural-language questions, and receive context-aware answers using Retrieval-Augmented Generation (RAG), local large language models, and semantic search.

Overview

The Agentic RAG Web Application combines a modern React frontend with a Python FastAPI backend. It uses Ollama to run a local language model and ChromaDB to store document embeddings for semantic retrieval.

The application processes uploaded documents, divides their text into manageable chunks, converts those chunks into embeddings, and retrieves relevant information when a user asks a question. The retrieved context is passed to the language model to generate an answer grounded in the available document content.

The project is designed to be extended with additional agent capabilities, including web search, SQL querying, and Model Context Protocol (MCP) tool integration.

 Features

- **Document Upload:** Upload PDF and TXT files through the web interface.
- **Document Processing:** Extract text and split it into smaller chunks.
- **Semantic Search:** Retrieve relevant document content using embeddings.
- **AI Question Answering:** Ask natural-language questions about uploaded documents.
- **Local LLM Integration:** Generate responses using Ollama and Llama 3.2.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Source Information:** Display document filenames associated with retrieved answers.
- **Document Management:** Support document listing, viewing, and deletion when those endpoints are enabled in the current version.
- **REST API:** Expose backend functionality through FastAPI.
- **Interactive API Documentation:** Test supported endpoints through Swagger UI.
- **Extensible Architecture:** Provide a foundation for adding agent routing and external tools.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React.js, Vite |
| Backend | Python, FastAPI |
| Language Model | Ollama, Llama 3.2 (3B) |
| Embedding Model | `nomic-embed-text` |
| Vector Database | ChromaDB |
| Document Processing | pypdf |
| HTTP Communication | HTTPX / Axios |
| API Testing | FastAPI Swagger UI |
| Version Control | Git and GitHub |

## System Architecture

```text
                   User
                    |
                    v
          React + Vite Frontend
                    |
                    v
             FastAPI Backend
                    |
          +---------+----------+
          |                    |
          v                    v
   Document Processing     User Question
          |                    |
          v                    v
     PDF / TXT Text      Query Embedding
          |                    |
          v                    v
     Text Chunking        ChromaDB Search
          |                    |
          v                    v
       Embeddings <---- Relevant Chunks
                               |
                               v
                         Ollama LLM
                               |
                               v
                       Generated Answer
                               |
                               v
                       React Frontend
```

## How It Works

### 1. Document Upload

The user uploads a PDF or TXT file using the React interface. The file is sent to the FastAPI backend for processing.

### 2. Text Extraction

The backend extracts text from PDF files using `pypdf` and reads text from TXT files.

### 3. Text Chunking

The extracted text is divided into smaller, overlapping chunks. Chunking helps the retrieval system find relevant portions of longer documents.

### 4. Embedding Generation

The embedding model converts document chunks into numerical vector representations that capture semantic meaning.

### 5. Vector Storage

The embeddings and associated document metadata are stored in ChromaDB for subsequent retrieval.

### 6. Question Processing

When a user asks a question, the backend converts the question into an embedding and searches ChromaDB for semantically relevant document chunks.

### 7. Answer Generation

The retrieved context and user question are sent to the local Ollama model. The model generates an answer based on the provided context.

### 8. Result Display

The frontend displays the generated answer and available source information.

## Project Structure

```text
Agentic-RAG/
|
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- agent_tools.py          # Optional agent tools module
|   |   |-- chroma_db/              # Generated local vector data
|   |   |-- uploads/                # Uploaded documents
|   |   |-- agent_data.db           # Optional SQLite database
|   |
|   |-- mcp_server.py               # Optional MCP server
|   |-- requirements.txt            # Backend dependencies
|   |
|-- frontend/
|   |-- src/
|   |   |-- App.jsx
|   |   |-- main.jsx
|   |   |-- index.css
|   |
|   |-- public/
|   |-- package.json
|   |-- vite.config.js
|
|-- .gitignore
|-- README.md
```

*Note: The structure above illustrates the main application and optional extension files. Keep only files and features that actually exist in your repository.*

## Prerequisites

Install the following before running the project:

- Python 3.11 or a compatible Python version supported by your installed dependencies.
- Node.js and npm.
- Ollama.
- Git.

Download links:

- Python: https://www.python.org/downloads/
- Node.js: https://nodejs.org/
- Ollama: https://ollama.com/
- Git: https://git-scm.com/

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/JyotiSadashivGudlaman/Agentic-RAG.git
cd Agentic-RAG
```

### Step 2: Set Up the Backend

Open a terminal in the project root and navigate to the backend:

```powershell
cd backend
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate the virtual environment in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

If `requirements.txt` has not yet been created, install the packages used by the current backend. For the original document RAG implementation, these may include:

```powershell
python -m pip install fastapi uvicorn httpx chromadb pypdf
```

Use the dependency versions validated by your working project when possible.

### Step 3: Set Up Ollama

Install and start Ollama, then download the chat model:

```powershell
ollama pull llama3.2:3b
```

Download the embedding model:

```powershell
ollama pull nomic-embed-text
```

Verify that both models are available:

```powershell
ollama list
```

Ollama normally provides its local API at:

```text
http://localhost:11434
```

### Step 4: Run the Backend

From the `backend` directory, run:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

The backend should be available at:

- API root: http://127.0.0.1:8001/
- Health endpoint: http://127.0.0.1:8001/health
- Swagger documentation: http://127.0.0.1:8001/docs

Keep the backend terminal open.

### Step 5: Set Up the Frontend

Open a second terminal:

```powershell
cd C:\Users\jyoti\Downloads\Agentic-RAG\frontend
```

Install frontend dependencies:

```powershell
npm install
```

Start the Vite development server:

```powershell
npm run dev
```

Open the local frontend URL displayed by Vite, typically:

http://localhost:5173/

Keep the frontend terminal open.

### Step 6: Verify the Application

1. Open the frontend.
2. Upload a PDF or TXT document.
3. Wait for document processing to complete.
4. Enter a question related to the document.
5. Submit the question.
6. Verify that an answer and available source information are displayed.

## API Endpoints

The following endpoints are part of the described backend implementation. Confirm the actual routes in `/docs`, because endpoint availability depends on your current `main.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Check backend status |
| GET | `/docs` | Interactive API documentation |
| GET | `/documents` | List indexed documents |
| POST | `/chat` | Send a direct prompt to the LLM |
| POST | `/upload` | Upload and index a document |
| POST | `/ask-rag` | Ask a question using document retrieval |
| GET | `/documents/file/{filename}` | View a stored document, if implemented |
| DELETE | `/documents/{filename}` | Delete a document, if implemented |

### Example: Upload a Document

The upload endpoint accepts a multipart form-data request containing a file.

```bash
curl -X POST "http://127.0.0.1:8001/upload" \
  -F "file=@example.pdf"
```

### Example: Ask a Question

```bash
curl -X POST "http://127.0.0.1:8001/ask-rag" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Summarize the document\"}"
```

## Configuration

The application uses the following local services and model names in the current setup:

| Setting | Value |
|---|---|
| Backend URL | `http://127.0.0.1:8001` |
| Frontend URL | `http://localhost:5173` |
| Ollama API | `http://localhost:11434` |
| Chat model | `llama3.2:3b` |
| Embedding model | `nomic-embed-text` |

If your frontend uses a Vite development proxy, verify that requests using `/api` are forwarded to the FastAPI backend.

## Troubleshooting

### 1. FastAPI Does Not Start

Check that your virtual environment is activated and the required packages are installed.

```powershell
python -m pip install -r requirements.txt
```

Ensure that you run Uvicorn from the `backend` directory using:

```powershell
python -m uvicorn app.main:app --reload --port 8001
```

### 2. Ollama Is Not Responding

Check the installed models:

```powershell
ollama list
```

Verify that the Ollama service is running and that the required models are installed.

### 3. The Frontend Cannot Reach the Backend

Check that FastAPI is running on port `8001` and inspect `frontend/vite.config.js` for the configured proxy.

### 4. No Relevant Document Is Retrieved

Verify that the document uploaded successfully and that text extraction returned content. Try a question containing specific terms from the document.

### 5. The `/upload` Endpoint Is Missing

Open:

http://127.0.0.1:8001/docs

If the endpoint is missing, verify that the correct FastAPI application module is running and that the upload route is defined in `backend/app/main.py`.

## Future Enhancements

The architecture can be extended with:

- LLM-based autonomous tool selection.
- Web search integration.
- Natural-language-to-SQL querying.
- MCP server and client integration.
- Multiple document formats.
- Streaming responses.
- Improved source citations and retrieval evaluation.
- Conversation history and memory.
- Authentication and user-specific document collections.
- Deployment using Docker and a cloud hosting platform.

These are planned enhancements unless they have been implemented and tested in the published version.

## Learning Outcomes

This project provides practical experience with:

- Full-stack web development using React and FastAPI.
- REST API development and frontend-backend integration.
- Local large language model integration.
- Text extraction and document preprocessing.
- Semantic search and vector databases.
- Embeddings and Retrieval-Augmented Generation.
- Git, GitHub, and software project organization.

## Author

**Jyoti Sadashiv Gudlamani**

GitHub: https://github.com/JyotiSadashivGudlaman
