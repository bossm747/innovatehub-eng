import React, { useState, useEffect } from "react";
import { CopilotKit } from "@copilotkit/react-core";
import { CopilotSidebar } from "@copilotkit/react-ui";
import logo from "../public/innovatehub-logo.png";

function formatTimestamp(date: Date) {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

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
      <div style={{ display: "flex", height: "100vh", background: "#f5f7fa" }}>
        <div style={{ flex: 1, padding: 32 }}>
          <header style={{ display: "flex", alignItems: "center", marginBottom: 24 }}>
            <img src={logo} alt="InnovateHub Logo" style={{ height: 48, marginRight: 16 }} />
            <h1 style={{ color: "#1a237e", fontWeight: 700, fontSize: 32 }}>InnovateHub AI App Developer</h1>
          </header>
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontWeight: 600, color: "#333" }}>Select Model: </label>
            {loading ? (
              <span style={{ marginLeft: 8 }}>Loading models...</span>
            ) : error ? (
              <span style={{ color: "red", marginLeft: 8 }}>{error}</span>
            ) : (
              <select
                value={selectedModel}
                onChange={e => setSelectedModel(e.target.value)}
                style={{ marginLeft: 8, padding: 4, borderRadius: 4 }}
              >
                {models.map((m, i) => (
                  <option key={i} value={m.model}>{m.provider} - {m.model}</option>
                ))}
              </select>
            )}
          </div>
          <div style={{ marginTop: 32, color: "#666" }}>
            <p>Welcome to <b>InnovateHub</b>! Select a model and start chatting with your AI developer assistant.</p>
          </div>
        </div>
        <div style={{ width: 420, background: "#fff", borderLeft: "1px solid #e0e0e0", display: "flex", flexDirection: "column" }}>
          <CopilotSidebar
            style={{ flex: 1 }}
            avatarUrl={logo}
            showTimestamps
            messageBubbleStyle={{ borderRadius: 16, padding: 12, margin: 8 }}
            userAvatarUrl="https://ui-avatars.com/api/?name=User&background=1a237e&color=fff"
            assistantAvatarUrl={logo}
            brandingFooter={<div style={{ textAlign: "center", padding: 12, color: "#888" }}>InnovateHub AI App Developer &copy; 2025</div>}
          />
        </div>
      </div>
    </CopilotKit>
  );
}

export default App;
