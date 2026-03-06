# Krishnamurti AI - Conversational Philosophy Teacher

Guides students through J. Krishnamurti's philosophy using RAG, state machine, and dynamic tone adaptation.

## System Architecture

**Pipeline**: User Input → Classifier → State Machine → RAG Retrieval → Generator → Response

```
User Message
    ↓
Classifier (GPT-4o-mini) → Classification (confusion/understanding/insight/minimal/off-topic)
    ↓
State Machine → Updates: Topic (4) | Stage (4) | Tone (4)
    ↓
RAG System → Retrieves 3 relevant passages from chunks (Krishnamurti texts)
    ↓
Generator (GPT-4o) → Krishnamurti-style response + updated state
    ↓
Frontend → Display response + update image/sidebar
```

## Core Components

**Backend** (`backend/`)
- `text_chunker.py` - Splits texts into 300-word chunks (50-word overlap)
- `rag_system.py` - ChromaDB vector DB + semantic search (all-MiniLM-L6-v2 embeddings)
- `state_machine.py` - Manages topic/stage/tone transitions
- `classifier.py` - Categorizes user input (GPT-4o-mini, temp=0)
- `generator.py` - Generates responses (GPT-4o, temp=0.8, max=400 tokens)

**Frontend** (`frontend/src/`)
- React app with real-time state updates and tone-based images

**State Machine**
- Topics: Suffering and its roots → Order and balance → Change and instability → Self and observer
- Stages: Introduction → Examination → Challenge → Resolution
- Tones: Teaching | Sad | Happy | Contemplate

## Parameter Choices

| Component | Parameter | Value | Why |
|-----------|-----------|-------|-----|
| **RAG** | Chunk Size | 300 words | Balance context vs precision |
| | Chunk Overlap | 50 words | Prevent boundary info loss |
| | Results | 3 passages | Sufficient context, not overwhelming |
| | Embeddings | all-MiniLM-L6-v2 | Fast, efficient (384-dim) |
| **Classifier** | Model | GPT-4o-mini | Fast, cheap, accurate |
| | Temperature | 0 | Deterministic classification |
| **Generator** | Model | GPT-4o | Better character consistency |
| | Temperature | 0.8 | Creative but not random |
| | Max Tokens | 400 | Concise (2-4 sentences) |
| **State** | Stage Threshold | 2-14 turns | Gradual progression |
| | Topic Threshold | 8+ turns | Deep engagement required |
| | History Window | 8 messages | Recent context for coherence |

## Setup

**Backend:**
```bash
source venv/bin/activate
pip install -r requirements.txt
python backend/rag_system.py  # Initialize DB
python app.py                 # Run server (port 5001)
```

**Frontend:**
```bash
cd frontend
npm install
npm start  # Opens http://localhost:3000
```

**Environment** (`.env`):
```
OPENAI_API_KEY=sk-...
CHROMA_API_KEY=...
CHROMA_TENANT=...
CHROMA_DATABASE=...
```

## Data Flow

1. Krishnamurti talks/books → text chunks
2. Chunks → 384-dim embeddings → ChromaDB
3. User input + topic keywords → semantic search → top 3 passages
4. Classification → state update → RAG retrieval → generation
5. Response (~3-5s) → frontend updates (image + sidebar)

## Key Design Choices

- **Two-step LLM**: Separate classification + generation for clarity
- **300-word chunks**: Sweet spot for context vs retrieval quality
- **GPT-4o-mini for classifier**: Speed + cost efficiency
- **GPT-4o for generator**: Character consistency matters
- **State-driven tone**: Image/teaching style adapts to user engagement
- **In-memory sessions**: Simple (use Redis for production)
