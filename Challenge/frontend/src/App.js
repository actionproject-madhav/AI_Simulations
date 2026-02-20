import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [state, setState] = useState(null);
  const [sessionId] = useState(`session_${Date.now()}`);
  const [initialized, setInitialized] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initialize conversation
  useEffect(() => {
    if (!initialized) {
      startConversation();
      setInitialized(true);
    }
  }, [initialized]);

  const startConversation = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });

      const data = await response.json();

      if (data.success) {
        setMessages([{ role: 'buddha', content: data.message }]);
        setState(data.state);
      } else {
        setMessages([{ role: 'system', content: `Error: ${data.error}` }]);
      }
    } catch (error) {
      setMessages([{ role: 'system', content: `Connection error: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();

    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        })
      });

      const data = await response.json();

      if (data.success) {
        setMessages(prev => [...prev, { role: 'buddha', content: data.message }]);
        setState(data.state);
      } else {
        setMessages(prev => [...prev, { role: 'system', content: `Error: ${data.error}` }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const resetConversation = async () => {
    setLoading(true);
    try {
      await fetch(`${API_BASE}/api/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });

      setMessages([]);
      setState(null);
      startConversation();
    } catch (error) {
      console.error('Reset error:', error);
    } finally {
      setLoading(false);
    }
  };

  const getImageForTone = (tone) => {
    const toneImages = {
      'compassionate': '/images/compassionate.jpg',
      'meditative': '/images/meditative.jpeg',
      'teaching': '/images/teaching.webp',
      'challenging': '/images/challenging.jpeg'
    };
    return toneImages[tone] || toneImages['compassionate'];
  };

  const topicNames = {
    'four_noble_truths': 'The Four Noble Truths',
    'middle_way': 'The Middle Way',
    'impermanence': 'Impermanence',
    'non_self': 'Non-Self'
  };

  return (
    <div className="app">
      <div className="sidebar">
        <div className="sidebar-header">
          <h1>Buddha</h1>
          <p className="subtitle">A Philosophical Dialogue</p>
        </div>

        {state && (
          <div className="buddha-image-container">
            <img
              src={getImageForTone(state.tone)}
              alt={`Buddha - ${state.tone}`}
              className="buddha-image"
            />
            <div className="image-label">{state.tone}</div>
          </div>
        )}

        {state && (
          <div className="state-info">
            <div className="state-item">
              <span className="state-label">Topic:</span>
              <span className="state-value">{topicNames[state.topic] || state.topic}</span>
            </div>
            <div className="state-item">
              <span className="state-label">Stage:</span>
              <span className="state-value">{state.stage}</span>
            </div>
            <div className="state-item">
              <span className="state-label">Turn:</span>
              <span className="state-value">{state.turn_count}</span>
            </div>
          </div>
        )}

        <button onClick={resetConversation} className="reset-button" disabled={loading}>
          Reset Conversation
        </button>

        <div className="sidebar-footer">
          <p>Respond thoughtfully.</p>
          <p>Buddha will guide you through core Buddhist concepts.</p>
        </div>
      </div>

      <div className="main-content">
        <div className="messages-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-label">
                {msg.role === 'user' ? 'You' : msg.role === 'buddha' ? 'Buddha' : 'System'}
              </div>
              <div className="message-content">
                {msg.content}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message buddha">
              <div className="message-label">Buddha</div>
              <div className="message-content loading">
                <span className="loading-dot">.</span>
                <span className="loading-dot">.</span>
                <span className="loading-dot">.</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={sendMessage} className="input-form">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your response..."
            disabled={loading}
            className="input-field"
          />
          <button type="submit" disabled={loading || !input.trim()} className="send-button">
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
