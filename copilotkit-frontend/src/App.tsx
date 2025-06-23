import React, { useState, useEffect } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";

function App() {
  const [models, setModels] = useState<{provider: string, model: string}[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/copilotkit/models")
      .then(res => res.json())
      .then(data => {
        setModels(data.models);
        setSelectedModel(data.models[0]?.model || "");
        setLoading(false);
      })
      .catch(e => {
        setError("Failed to load models");
        setLoading(false);
      });
  }, []);

  return (
    <CopilotKit runtimeUrl="/api/copilotkit">
      <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", background: "#f6f8fa" }}>
        <header style={{ display: "flex", alignItems: "center", background: "#1e293b", color: "#fff", padding: "16px 32px" }}>
          <img src={process.env.PUBLIC_URL + "/innovatehub-logo.png"} alt="Innovate Hub Logo" style={{ height: 48, marginRight: 20, borderRadius: 8, background: "#fff" }} />
          <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0 }}>InnovateHub AI App Developer</h1>
        </header>
        <div style={{ display: "flex", flex: 1 }}>
          <div style={{ flex: 1, padding: 32 }}>
            <h2 style={{ color: "#1e293b" }}>Welcome!</h2>
            {loading ? (
              <p>Loading models...</p>
            ) : error ? (
              <p style={{ color: 'red' }}>{error}</p>
            ) : (
              <label>
                <b>Select Model:</b>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  style={{ marginLeft: 8, padding: 4, borderRadius: 4, border: '1px solid #cbd5e1' }}
                >
                  {models.map(m => (
                    <option key={m.model} value={m.model}>
                      {m.provider} - {m.model}
                    </option>
                  ))}
                </select>
              </label>
            )}
            <p style={{ marginTop: 16, color: "#475569" }}>
              Start chatting with your selected LLM in the sidebar!
            </p>
          </div>
          <CopilotSidebar
            defaultOpen={true}
            labels={{
              title: "InnovateHub AI Assistant",
              initial: "Hello! How can I help you today?",
            }}
            context={{ model: selectedModel }}
          />
        </div>
        <footer style={{ background: "#1e293b", color: "#fff", textAlign: "center", padding: 12 }}>
          Powered by Innovate Hub &bull; Multi-LLM AG UI Demo
        </footer>
      </div>
    </CopilotKit>
  );
}

export default App;
