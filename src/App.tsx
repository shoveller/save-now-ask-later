import {useAgent} from "agents/react";
import {useAgentChat} from "agents/chat/react";
import {useState} from 'react';
import './App.css';

function App() {
  const agent = useAgent({ agent: 'RagAgent' })
  const { messages, sendMessage, status, error, connectionError } = useAgentChat({ agent })
  const [input, setInput] = useState('')
  const [sendError, setSendError] = useState('')
  const busy = status === 'submitted' || status === 'streaming'

  return (
    <main className="chat">
      <header>
        <h1>Save Now, Ask Later</h1>
        <p>Ask to save a URL, search your saved pages, or list your sources.</p>
      </header>
      <section className="messages" aria-label="Conversation" aria-live="polite">
        {messages.map(message => (
          <article key={message.id} className="message">
            <strong>{message.role === 'user' ? 'You' : 'Assistant'}</strong>
            {message.parts.map((part, index) => part.type === 'text'
              ? <p key={index}>{part.text}</p>
              : part.type.startsWith('tool-') || part.type === 'dynamic-tool'
                ? <small key={index}>Tool: {part.type}</small>
                : null)}
          </article>
        ))}
        {busy && <p role="status">Working...</p>}
      </section>
      {(error || connectionError || sendError) &&
        <p role="alert">{sendError || error?.message || connectionError?.message}</p>}
      <form onSubmit={async event => {
        event.preventDefault()
        const text = input.trim()
        if (!text || busy) return
        setSendError('')
        try {
          await sendMessage({text})
          setInput('')
        } catch {
          setSendError('Message could not be sent. Please try again.')
        }
      }}>
        <label htmlFor="message">Message</label>
        <textarea id="message" value={input} onChange={event => setInput(event.target.value)}
          placeholder="Save https://example.com" disabled={busy} rows={3} />
        <button type="submit" disabled={busy || !input.trim()}>Send</button>
      </form>
    </main>
  )
}

export default App
