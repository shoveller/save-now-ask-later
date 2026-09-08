import {useAgent} from "agents/react";
import {useAgentChat} from "agents/chat/react";

function App() {
  const agent = useAgent({ agent: 'RagAgent' })
  const { messages } = useAgentChat({ agent })

  return (
    <>{messages}</>
  )
}

export default App
