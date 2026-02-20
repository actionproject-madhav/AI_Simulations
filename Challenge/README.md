# Buddha AI - A Philosophical Dialogue System

A conversational AI application that simulates a dialogue with Buddha (Siddhartha Gautama), guiding students through Buddhist philosophy using RAG (Retrieval-Augmented Generation), state machine-driven conversation flow, and dynamic tone adaptation.

## Project Structure

```
Challenge/
├── backend/                  # Python backend components
│   ├── text_chunker.py      # Text chunking for RAG
│   ├── rag_system.py        # Vector database (ChromaDB)
│   ├── state_machine.py     # Conversation state management
│   ├── classifier.py        # User input classification (LLM)
│   ├── generator.py         # Response generation (LLM)
│   └── __init__.py
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Classic black/white styling
│   │   ├── index.js
│   │   └── index.css
│   ├── public/
│   │   └── index.html
│   └── package.json
├── texts/                    # Buddhist texts (115K words)
│   ├── Dhammapada.txt
│   └── sayings.txt
├── static/images/            # Buddha images for different tones
│   ├── compassionate.jpg
│   ├── meditative.jpeg
│   ├── teaching.webp
│   └── challenging.jpeg
├── data/                     # ChromaDB vector database (generated)
├── venv/                     # Python virtual environment
├── app.py                    # Flask backend server
├── config.py                 # Configuration
├── .env                      # Environment variables (YOUR API KEYS)
├── .gitignore
├── requirements.txt
└── README.md                 # This file
```

## Features

### Backend (Python + Flask)
- **RAG System**: 462 text chunks from Buddhist texts, semantic search via ChromaDB
- **State Machine**: Tracks topic (4 concepts), stage (4 stages), and tone (4 tones)
- **LLM Classifier**: Categorizes user responses into 7 types
- **Response Generator**: Creates Buddha's responses using GPT-4o with character consistency
- **RESTful API**: `/api/start`, `/api/chat`, `/api/state`, `/api/reset`

### Frontend (React)
- **Classic Design**: Black/white color scheme, no gradients, serif typography
- **Reactive UI**: Real-time state updates, dynamic image changes based on tone
- **Responsive**: Works on desktop and mobile
- **Smooth UX**: Auto-scrolling, loading indicators, error handling

### Philosophical System
**Topics** (Progressive difficulty):
1. Four Noble Truths (Foundation)
2. The Middle Way (Balance)
3. Impermanence (Anicca)
4. Non-Self (Anatta) - Most challenging

**Stages** (Per topic):
- Introduction → Examination → Challenge → Resolution

**Tones** (Dynamic adaptation):
- Compassionate (confusion/struggle)
- Meditative (deep contemplation)
- Teaching (active dialogue)
- Challenging (shallow answers/overconfidence)

## Setup Instructions

### 1. Environment Variables

Create/edit the `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-proj-...

# Optional (has defaults)
OPENAI_MODEL_CLASSIFIER=gpt-4o-mini
OPENAI_MODEL_GENERATOR=gpt-4o
DEBUG=True
PORT=5000
```

**IMPORTANT**: Make sure your `.env` file format is correct:
```
OPENAI_API_KEY=your_actual_key_here
```
(No quotes, no spaces around `=`)

### 2. Backend Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install Python dependencies (if not already installed)
pip install -r requirements.txt

# Test individual components
python backend/text_chunker.py      # Should create 462 chunks
python backend/rag_system.py         # Should build vector DB
python backend/state_machine.py      # Should show state transitions

# Test with API key
python config.py                     # Should show "Valid: True"
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Return to root directory
cd ..
```

## Running the Application

### Option A: Development Mode (Recommended for testing)

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
python app.py
```
Backend runs on `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```
Frontend runs on `http://localhost:3000`

The React dev server will proxy API requests to the Flask backend.

### Option B: Production Mode

```bash
# Build React frontend
cd frontend
npm run build
cd ..

# Run Flask (serves both API and built React app)
source venv/bin/activate
python app.py
```

Visit `http://localhost:5000`

## Testing the System

### 1. Test Backend API

```bash
# Start Flask server
source venv/bin/activate
python app.py

# In another terminal, test endpoints:
curl http://localhost:5000/api/health

curl -X POST http://localhost:5000/api/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test"}'

curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "What is suffering?"}'
```

### 2. Test Frontend

Open `http://localhost:3000` and try:
- Expressing confusion: "I don't understand"
- Showing understanding: "So attachment causes suffering?"
- Being insightful: "The self is just changing phenomena!"
- Being minimal: "ok" or "yeah"
- Going off-topic: "What about cryptocurrency?"

Watch how Buddha's tone and image change based on your responses.

### 3. Test State Transitions

Have a full conversation and observe:
- Topic progression: Four Noble Truths → Middle Way → Impermanence → Non-Self
- Stage progression: Introduction → Examination → Challenge → Resolution
- Tone changes: Based on your engagement quality

## Troubleshooting

### "OPENAI_API_KEY not found"
- Check `.env` file exists in root directory
- Verify format: `OPENAI_API_KEY=sk-...` (no quotes, no spaces)
- Make sure you're running from the Challenge/ directory

### "ModuleNotFoundError"
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### ChromaDB errors
```bash
rm -rf data/chroma
python backend/rag_system.py  # Rebuilds database
```

### CORS errors
- Make sure Flask backend is running on port 5000
- Check `frontend/package.json` has `"proxy": "http://localhost:5000"`

## Architecture

### Request Flow

```
User Input
    ↓
Frontend (React)
    ↓
Flask API (/api/chat)
    ↓
1. Classifier (GPT-4o-mini) → Categories: understanding/confusion/insight/etc.
    ↓
2. State Machine → Updates topic/stage/tone
    ↓
3. RAG System (ChromaDB) → Retrieves relevant Buddhist text passages
    ↓
4. Generator (GPT-4o) → Creates Buddha's response with:
    - Current state (topic, stage, tone)
    - RAG context
    - Character design
    - Conversation history
    ↓
Response to Frontend
    ↓
UI Updates (message + image change)
```

### Key Design Decisions

1. **Two-Step LLM Pipeline**: Separate classification and generation calls (as required)
2. **Chunk Size**: 300 words with 50-word overlap (balances context and retrieval precision)
3. **Model Selection**:
   - gpt-4o-mini for classification (fast, cheap, accurate for simple task)
   - gpt-4o for generation (higher quality responses, better character consistency)
4. **State Persistence**: In-memory dict (for demo); use Redis/DB for production
5. **Frontend Styling**: Serif fonts, black/white only, no emojis, no gradients (philosophical aesthetic)

## File Descriptions

### Backend
- `backend/text_chunker.py`: Loads and chunks Buddhist texts
- `backend/rag_system.py`: Embeddings + ChromaDB vector database
- `backend/state_machine.py`: Topic/stage/tone state management
- `backend/classifier.py`: Classifies user inputs via OpenAI API
- `backend/generator.py`: Generates Buddha's responses via OpenAI API
- `app.py`: Flask server with API endpoints
- `config.py`: Configuration and environment variables

### Frontend
- `frontend/src/App.js`: Main React component (conversation UI)
- `frontend/src/App.css`: Classic black/white styling
- `frontend/src/index.js`: React entry point

### Design Documents
- `character_design.md`: Buddha's personality, rhetorical style (1,100 words)
- `topic_graph.md`: 4 topics with transitions and teaching approach
- `tone_design.md`: 4 tones with linguistic characteristics

## Performance & Costs

### Processing
- Vector DB initialization: ~30 seconds (one-time)
- Classification: ~0.5-1 second per turn
- Response generation: ~2-4 seconds per turn
- Total response time: ~3-5 seconds

### API Costs (Estimated)
- Classification: ~$0.001 per turn (gpt-4o-mini)
- Generation: ~$0.01-0.02 per turn (gpt-4o)
- Full conversation (20 turns): ~$0.20-0.40

## Development Notes

### Adding a New Topic
1. Update `backend/state_machine.py` TOPICS dict
2. Update `backend/rag_system.py` topic_keywords mapping
3. Update `backend/generator.py` topic_intros
4. Update `character_design.md` and `topic_graph.md`

### Changing Tone Behavior
1. Update `backend/state_machine.py` TONE_TRANSITIONS
2. Update `backend/generator.py` tone_instructions
3. Add/change images in `static/images/`
4. Update `frontend/src/App.js` getImageForTone()

### Adjusting Difficulty
- Chunk size: `config.py` RAG_CHUNK_SIZE
- Stage progression speed: `backend/state_machine.py` update_stage()
- Topic transition threshold: `backend/state_machine.py` should_transition_topic()

## Credits

**Texts**: Public domain Buddhist texts (Dhammapada, Buddhist Sayings)
**Technology**: OpenAI GPT-4o, ChromaDB, Sentence Transformers, React, Flask
**Character Design**: Interpretation based on textual evidence from Buddhist literature

## License

Educational project for CMS.636 course.
