import { askQuestion } from './api'
import Win95Chat, { type AskResult } from './Win95Chat'

async function handleAsk(question: string): Promise<AskResult> {
  const result = await askQuestion(question)
  return {
    answer: result.answer,
    sources: result.cited_sections.map((s) => ({ title: `§${s}` })),
  }
}

function App() {
  return (
    <Win95Chat
      onAsk={handleAsk}
      title="NYC Building Code Assistant"
      greeting="Ask me anything about the 2022 NYC Building Code."
    />
  )
}

export default App
