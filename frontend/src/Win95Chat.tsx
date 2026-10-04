import { useEffect, useRef, useState, type FormEvent } from 'react'
import './win95.css'

export interface ChatSource {
  title: string
  url?: string
}

export type AskResult = string | { answer: string; sources?: ChatSource[] }

interface ChatMessage {
  role: 'bot' | 'user'
  text: string
  sources?: ChatSource[]
}

interface Win95ChatProps {
  onAsk: (question: string) => Promise<AskResult>
  title?: string
  greeting?: string
}

/**
 * Windows 95-styled chat window for the RAG app.
 */
export default function Win95Chat({
  onAsk,
  title = 'Ask the Docs',
  greeting = 'Hi! Ask me anything about your documents.',
}: Win95ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([{ role: 'bot', text: greeting }])
  const [draft, setDraft] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const clock = useClock()

  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, loading])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const q = draft.trim()
    if (!q || loading) return
    setDraft('')
    setError(null)
    setMessages((m) => [...m, { role: 'user', text: q }])
    setLoading(true)
    try {
      const res = await onAsk(q)
      const answer = typeof res === 'string' ? res : res?.answer ?? ''
      const sources = typeof res === 'string' ? [] : res?.sources ?? []
      setMessages((m) => [...m, { role: 'bot', text: answer, sources }])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="w95-desktop">
      <div className="w95-stage">
        <section className="w95-window w95-raised" aria-label={title}>
          <header className="w95-titlebar">
            <div className="w95-title">
              <ChatIcon />
              <span>{title}</span>
            </div>
            <div className="w95-title-buttons" aria-hidden="true">
              <span className="w95-btn w95-tbtn w95-raised">
                <span style={{ width: 7, height: 2, background: '#000', alignSelf: 'flex-end', marginBottom: 3 }} />
              </span>
              <span className="w95-btn w95-tbtn w95-raised">
                <span style={{ width: 7, height: 6, border: '1px solid #000', borderTopWidth: 2 }} />
              </span>
              <span className="w95-btn w95-tbtn w95-raised" style={{ marginLeft: 2 }}>✕</span>
            </div>
          </header>

          <nav className="w95-menubar" aria-hidden="true">
            <span><u>F</u>ile</span>
            <span><u>E</u>dit</span>
            <span><u>H</u>elp</span>
          </nav>

          <div className="w95-body">
            <div ref={listRef} className="w95-messages w95-sunken" role="log" aria-live="polite">
              {messages.map((m, i) => (
                <div key={i} className={`w95-msg ${m.role}`}>
                  <div className="w95-who">{m.role === 'user' ? 'You:' : 'Assistant:'}</div>
                  <div className="w95-text">{m.text}</div>
                  {m.sources && m.sources.length > 0 && (
                    <div className="w95-source">
                      Source:{' '}
                      {m.sources.map((s, j) => (
                        <span key={j}>
                          {j > 0 && ', '}
                          {s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {loading && <div className="w95-thinking">Searching documents…</div>}
            </div>

            <form className="w95-form" onSubmit={handleSubmit}>
              <label htmlFor="w95-q" className="w95-sr-only">Ask a question</label>
              <input
                id="w95-q"
                className="w95-input w95-sunken"
                type="text"
                placeholder="Type your question…"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                disabled={loading}
                autoComplete="off"
              />
              <button type="submit" className="w95-btn w95-send w95-raised" disabled={loading || !draft.trim()}>
                <u>S</u>end
              </button>
            </form>
          </div>

          <footer className="w95-statusbar">
            <div className="w95-sunken">{error ? `Error: ${error}` : loading ? 'Working…' : 'Ready'}</div>
            <div className="w95-sunken">Connected</div>
          </footer>
        </section>
      </div>

      <div className="w95-taskbar w95-raised">
        <div className="w95-taskbar-left">
          <button type="button" className="w95-btn w95-start w95-raised">
            <StartIcon /> Start
          </button>
          <div className="w95-task w95-sunken">{title}</div>
        </div>
        <div className="w95-clock w95-sunken">{clock}</div>
      </div>
    </div>
  )
}

function useClock() {
  const fmt = () => new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  const [t, setT] = useState(fmt)
  useEffect(() => {
    const id = setInterval(() => setT(fmt()), 15000)
    return () => clearInterval(id)
  }, [])
  return t
}

function ChatIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <rect x="1" y="2" width="14" height="10" fill="#fff" stroke="#000" />
      <rect x="3" y="4" width="8" height="1" fill="#000080" />
      <rect x="3" y="7" width="6" height="1" fill="#000080" />
      <path d="M4 12 L4 15 L7 12 Z" fill="#fff" stroke="#000" />
    </svg>
  )
}

function StartIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
      <rect x="0" y="0" width="6" height="6" fill="#ff0000" />
      <rect x="7" y="0" width="6" height="6" fill="#00a000" />
      <rect x="0" y="7" width="6" height="6" fill="#0000ff" />
      <rect x="7" y="7" width="6" height="6" fill="#ffd700" />
    </svg>
  )
}
