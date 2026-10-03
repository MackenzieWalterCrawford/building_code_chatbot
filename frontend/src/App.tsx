import { useState, type FormEvent } from 'react'
import { askQuestion, type AskResponse } from './api'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setLoading(true)
    setError(null)
    setAnswer(null)

    try {
      setAnswer(await askQuestion(trimmed))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <h1>NYC Building Code Assistant</h1>
      <p className="subtitle">Ask a question about the 2022 NYC Building Code.</p>

      <form onSubmit={handleSubmit} className="question-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. What is the minimum floor live load for office occupancies?"
          rows={3}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? 'Asking…' : 'Ask'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {answer && (
        <section className={`answer${answer.insufficient ? ' insufficient' : ''}`}>
          <p>{answer.answer}</p>
          {answer.cited_sections.length > 0 && (
            <p className="citations">
              Cited: {answer.cited_sections.map((s) => `§${s}`).join(', ')}
            </p>
          )}
        </section>
      )}
    </main>
  )
}

export default App
