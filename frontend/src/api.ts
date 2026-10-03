export interface SourceChunk {
  section_number: string
  section_title: string
  chapter: string
  chapter_title: string
  text: string
  page_start: number
  page_end: number
  score: number
}

export interface AskResponse {
  answer: string
  sources: SourceChunk[]
  cited_sections: string[]
  confidence: number
  insufficient: boolean
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Request failed (${res.status}): ${detail}`)
  }

  return res.json()
}
