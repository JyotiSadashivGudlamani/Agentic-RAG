
import { useState } from 'react'
import axios from 'axios'
import './App.css'

const API = '/api'

function App() {
  const [file, setFile] = useState(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')

  async function uploadDocument(event) {
    event.preventDefault()

    if (!file) {
      setError('Please select a PDF or TXT file.')
      return
    }

    const data = new FormData()
    data.append('file', file)

    setLoading(true)
    setError('')
    setStatus('Uploading and processing your document...')

    try {
      const response = await axios.post(`${API}/upload`, data)
      setStatus(
        response.data.message ||
        `Upload successful: ${file.name}`
      )
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Upload failed. Check that the backend is running.'
      )
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  async function askQuestion(event) {
    event.preventDefault()

    if (!question.trim()) {
      setError('Please enter a question.')
      return
    }

    setLoading(true)
    setError('')
    setAnswer('')
    setSources([])

    try {
      const response = await axios.post(`${API}/ask-rag`, {
        question: question.trim(),
      })

      setAnswer(response.data.answer || 'No answer returned.')
      setSources(response.data.sources || [])
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        'Could not get an answer. Check the backend and request format.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">✦</div>
          <div>
            <h1>RAG Assistant</h1>
            <p>Chat with your documents</p>
          </div>
        </div>
        <span className="connection">Local AI</span>
      </header>

      <main className="main">
        <section className="hero">
          <span className="eyebrow">YOUR DOCUMENT WORKSPACE</span>
          <h2>Knowledge, made searchable.</h2>
          <p>
            Upload your documents and ask questions in plain English.
            Your assistant retrieves relevant content before generating
            an answer.
          </p>
        </section>

        <div className="workspace">
          <section className="panel upload-panel">
            <div className="panel-heading">
              <span className="panel-icon">↑</span>
              <div>
                <h3>Upload a document</h3>
                <p>PDF or TXT format</p>
              </div>
            </div>

            <form onSubmit={uploadDocument}>
              <label className="dropzone">
                <span className="upload-symbol">⇧</span>
                <strong>
                  {file ? file.name : 'Choose a file'}
                </strong>
                <span>Click here to browse your computer</span>
                <input
                  type="file"
                  accept=".pdf,.txt,application/pdf,text/plain"
                  onChange={(event) =>
                    setFile(event.target.files?.[0] || null)
                  }
                />
              </label>

              <button className="primary-button" disabled={loading}>
                {loading ? 'Please wait...' : 'Upload document'}
              </button>
            </form>

            <div className="tip">
              <strong>How it works</strong>
              <p>
                Text is extracted, split into chunks, embedded, and
                stored in your vector database.
              </p>
            </div>
          </section>

          <section className="panel chat-panel">
            <div className="panel-heading">
              <span className="panel-icon">✧</span>
              <div>
                <h3>Ask your documents</h3>
                <p>Answers grounded in retrieved content</p>
              </div>
            </div>

            <form onSubmit={askQuestion}>
              <label className="field-label" htmlFor="question">
                Your question
              </label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Summarize the main topics in my PDF..."
                rows={4}
              />
              <button className="primary-button" disabled={loading}>
                {loading ? 'Thinking...' : 'Ask question →'}
              </button>
            </form>

            <div className="answer-area">
              <h4>Answer</h4>
              {answer ? (
                <div className="answer-text">{answer}</div>
              ) : (
                <p className="placeholder">
                  Your answer will appear here after you ask a question.
                </p>
              )}

              {sources.length > 0 && (
                <div className="sources">
                  <h4>Sources</h4>
                  {sources.map((source, index) => (
                    <span className="source-chip" key={`${source}-${index}`}>
                      ▤ {source}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </section>
        </div>

        {status && <div className="notice success">{status}</div>}
        {error && <div className="notice failure">{error}</div>}

        <footer>
          React + Vite · FastAPI · Ollama · ChromaDB
        </footer>
      </main>
    </div>
  )
}

export default App
