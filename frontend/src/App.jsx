import React, { useState, useRef, useEffect } from 'react';
import { Send, MapPin, FileText, Shield, Info, Terminal, ChevronRight, Sparkles, Cpu } from 'lucide-react';
import ConfigPanel from './components/ConfigPanel';
import ThinkingTrace from './components/ThinkingTrace';

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [history, setHistory] = useState([]);
  const [config, setConfig] = useState({
    model: 'llama3.1:8b',
    temperature: 0.1,
    top_k: 3
  });

  const chatEndRef = useRef(null);
  const scrollToBottom = () => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(scrollToBottom, [history]);

  const handleSend = async () => {
    if (!query.trim() || loading) return;

    const userMsg = { role: 'user', content: query };
    setHistory([...history, userMsg]);
    setLoading(true);
    setResponse(null);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, config }),
      });

      if (!res.ok) throw new Error('API request failed');

      const data = await res.json();
      setResponse(data);
      setHistory(prev => [...prev, { role: 'assistant', content: data.answer, data }]);
    } catch (err) {
      setHistory(prev => [...prev, { role: 'assistant', content: "Error: " + err.message, isError: true }]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  return (
    <div className="flex h-screen bg-bg-deep text-text-main overflow-hidden">
      {/* Sidebar - Config */}
      <aside className="w-30-pct border-r border-glass bg-bg-surface flex flex-col">
        <div className="p-8 border-b border-glass">
          <div className="flex items-center gap-3 mb-1">
            <MapPin className="text-primary" size={24} />
            <h1 className="text-xl font-bold tracking-tight text-white">WanderOn</h1>
          </div>
          <p className="text-[10px] text-text-muted uppercase tracking-[0.2em] font-bold ml-1">Travel Logic Engine</p>
        </div>

        <ConfigPanel config={config} setConfig={setConfig} />

        <div className="mt-auto p-6 bg-gradient-to-t from-bg-deep to-transparent">
          <div className="glass-card p-4 text-[10px] flex items-center gap-2">
            <Shield className="text-accent" size={14} />
            <span className="text-text-dim ">Production AI Guardrails Enabled</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="w-70-pct flex flex-col relative bg-[url('https://www.transparenttextures.com/patterns/dark-matter.png')]">
        <div className="chat-container">
          <div className="content-container flex-1 flex flex-col gap-8">
            {history.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center max-w-2xl mx-auto text-center space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
                <div className="flex items-center gap-4 mb-4">
                  <div className="w-12 h-12 rounded-2xl bg-surface border border-glass flex items-center justify-center">
                    <Info size={24} className="text-primary opacity-50" />
                  </div>
                  <h3 className="text-white font-medium">How can I help with your travel?</h3>
                </div>
                <p className="text-text-dim text-lg mb-2">Ask about visa requirements, per-diem rates, flight policies, or company reimbursement rules.</p>
                <div className="grid grid-cols-2 gap-4 w-full">
                  {['Do I need a visa to travel from India to London?', 'What is the per-diem rate in Mumbai?', 'What is the flight booking policy for international travel?', 'What is the international trip approval process?'].map(suggest => (
                    <button
                      key={suggest}
                      onClick={() => setQuery(suggest)}
                      className="glass-card p-4 text-sm text-text-dim hover:text-white hover:border-primary hover:bg-white/5 hover:scale-[1.02] transition-all text-left flex items-center justify-between group"
                    >
                      <span>{suggest}</span>
                      <ChevronRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity text-primary" />
                    </button>
                  ))}
                </div>
              </div>
            )}

            {history.map((msg, i) => (
              <div key={i} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-500`}>
                {msg.role === 'user' ? (
                  <div className="max-w-[70%] p-4 px-6 rounded-2xl bg-primary text-white shadow-xl shadow-primary-glow/20">
                    <p className="text-sm leading-relaxed">{msg.content}</p>
                  </div>
                ) : (
                  <div className={`max-w-[85%] w-full flex gap-4 ${msg.isError ? 'text-error' : ''}`}>
                    <div className="w-8 h-8 rounded-lg bg-surface border border-glass flex items-center justify-center flex-shrink-0 mt-1">
                      <Sparkles size={16} className="text-accent" />
                    </div>
                    <div className="flex-1 space-y-6">
                      <div className="text-base leading-relaxed text-text-main prose prose-invert overflow-hidden">
                        {msg.content}
                      </div>

                      {msg.data && msg.data.total_tokens !== undefined && (
                        <div className="flex items-center gap-1.9 mt-2 bg-white/5 border border-glass w-fit px-2 py-0.5 rounded-md">
                          <Cpu size={10} className="text-text-dim" />
                          <span style={{ fontSize: '10px', color: 'rgb(206, 202, 202)', letterSpacing: '0.02em', paddingLeft: '2px' }}>
                            {msg.data.total_tokens} tokens
                          </span>
                        </div>
                      )}

                      {msg.data && msg.data.sources && msg.data.sources.length > 0 && (
                        <div className="pt-4 border-t border-glass/50 grid gap-3">
                          <p className="text-[10px] font-bold text-text-muted uppercase tracking-wider flex items-center gap-2">
                            <FileText size={12} /> Cited Sources
                          </p>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {msg.data.sources.map((s, si) => (
                              <div key={si} className="text-xs bg-bg-surface/50 rounded-xl p-4 border border-glass border-l-primary border-l-2 hover:bg-bg-surface transition-colors">
                                <span className="font-bold text-primary block mb-1">{s.doc_id}</span>
                                <span className="text-text-dim italic">"{s.text_snippet}..."</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="input-area pb-12">
          <div className="content-container">
            <div className="input-wrapper">
              <textarea
                placeholder="Ask a travel policy question..."
                style={{ height: '64px' }}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />
              <button
                className={`btn-send ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
                style={{ marginBottom: '8px' }}
                onClick={handleSend}
                disabled={loading}
              >
                {loading ? <Terminal size={18} className="animate-spin" /> : <Send size={18} />}
              </button>
            </div>
            <p className="text-center text-[10px] text-text-muted py-3 border-b border-glass mb-2" style={{
              fontSize: '11px',
              color: 'rgba(206, 202, 202, 0.6)',
              textAlign: 'center',
            }}>
              Powered by local LLM • Grounded in FAISS Vector Store • WanderOn Agent v1.0
            </p>

            {/* Thinking Trace repositioned here to match input width */}
            <div className="max-w-4xl mx-auto px-4 py-6">
              <ThinkingTrace trace={loading ? [{ event: 'PROCESSING', status: 'warning', message: 'The agent is thinking...', timestamp: new Date().toISOString() }] : response?.trace} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
