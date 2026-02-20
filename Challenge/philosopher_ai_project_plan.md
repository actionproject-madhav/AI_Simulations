# Philosopher AI Project - Complete Implementation Plan

## PHASE 1: RESEARCH AND CONTENT PREPARATION (DO FIRST)

### What to Download:
1. **Philosopher's Texts** (at least 10,000 words total)
   - Source: Project Gutenberg (https://www.gutenberg.org/) or Internet Classics Archive (http://classics.mit.edu/)
   - Suggested philosophers:
     - **Plato** - The Republic, Apology, Phaedo (easy to find, rich character)
     - **Nietzsche** - Beyond Good and Evil, Thus Spoke Zarathustra
     - **Kant** - Critique of Pure Reason (dense but iconic)
     - **Mary Wollstonecraft** - A Vindication of the Rights of Woman
     - **Marcus Aurelius** - Meditations
   - Format: Download as .txt files (plain text preferred for parsing)
   - Save to: `texts/` directory

2. **Images for Character** (optional but recommended)
   - If using AI generation: Generate 4+ images representing different tones
   - If using other art: Find/create images for each tone state
   - Save to: `static/images/` directory

### What to Prepare:
1. **Character Design Document** (~750 words)
   - Read your chosen texts first (at least skim major sections)
   - Identify 4+ major concepts/topics from the works
   - Document:
     - Personality traits
     - Rhetorical style (how they argue, ask questions)
     - Values and priorities
     - How they respond to: confusion, insight, ignorance
     - Textual evidence for each choice
   - Save as: `character_design.md`

2. **Topic Graph Design**
   - Identify 4+ philosophical concepts from the texts
   - Map relationships between topics (which lead to which)
   - Example for Plato:
     - Justice → The Forms → The Cave Allegory → The Philosopher King
   - Save as: `topic_graph.md`

3. **Tone States Design**
   - Define 4+ tone states (e.g., warm, probing, skeptical, playful)
   - Describe how each tone manifests in language
   - Define transition triggers (what causes tone changes)
   - Save as: `tone_design.md`

---

## PHASE 2: ENVIRONMENT SETUP AND DEPENDENCIES

### What to Set Up:

1. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Core Dependencies**
   ```bash
   pip install chromadb
   pip install sentence-transformers
   pip install openai
   pip install python-dotenv
   pip install flask  # for web server
   pip install flask-cors  # for API calls from frontend
   ```

3. **OpenAI API Key Management**
   - Wait for professor's email with API key
   - Create `.env` file in project root:
     ```
     OPENAI_API_KEY=your_key_here
     ```
   - Add `.env` to `.gitignore` (CRITICAL - don't upload key!)

4. **Project Directory Structure**
   ```
   philosopher_ai/
   ├── .env                    # API key (don't commit!)
   ├── .gitignore             # Include .env, venv/
   ├── texts/                 # Downloaded philosopher texts
   ├── data/                  # Vector database storage
   ├── static/
   │   ├── images/           # Character images
   │   ├── css/              # Styles
   │   └── js/               # Frontend JavaScript
   ├── templates/
   │   └── index.html        # Main page
   ├── backend/
   │   ├── rag_system.py     # Vector DB and retrieval
   │   ├── state_machine.py  # State management
   │   ├── classifier.py     # User input classification
   │   ├── generator.py      # Response generation
   │   └── text_chunker.py   # Text preprocessing
   ├── app.py                # Flask server
   ├── config.py             # Configuration
   ├── character_design.md   # Your character document
   └── requirements.txt      # Dependencies list
   ```

5. **Create Requirements File**
   ```bash
   pip freeze > requirements.txt
   ```

---

## PHASE 3: RAG SYSTEM IMPLEMENTATION

### Research First:
- Read about chunk sizing (typically 200-500 words, but test different sizes)
- Understand sentence-transformers models (e.g., 'all-MiniLM-L6-v2')
- Learn ChromaDB basics

### What to Code:

1. **Text Chunker** (`text_chunker.py`)
   - Load text files from `texts/` directory
   - Split into chunks (experiment with size: 200, 300, 500 words)
   - Preserve context (overlap between chunks)
   - Associate chunks with metadata (source, topic, etc.)

2. **Vector Database Setup** (`rag_system.py`)
   - Initialize ChromaDB client
   - Load sentence-transformers model
   - Encode all text chunks
   - Store in ChromaDB with metadata
   - Implement query function: `get_relevant_passages(query, topic, n=3)`

3. **Build Minimal Test**
   - Create test script to verify:
     - Texts load correctly
     - Chunks are reasonable size
     - Queries return relevant passages
   - Example: Query "What is justice?" should return relevant Plato passages

---

## PHASE 4: STATE MACHINE DESIGN AND IMPLEMENTATION

### What to Code:

1. **State Model** (`state_machine.py`)
   - Define State class with:
     - `topic` (current concept being discussed)
     - `stage` (Introduction/Examination/Challenge/Resolution)
     - `tone` (warm/probing/skeptical/playful/etc.)
   - Define topic graph structure
   - Implement transition logic:
     ```python
     def transition(current_state, classification, context):
         # Determine new topic/stage/tone based on user input classification
         return new_state
     ```

2. **Stage Progression Logic**
   - Introduction → Examination (after initial engagement)
   - Examination → Challenge (after user shows basic understanding)
   - Challenge → Resolution (after thoughtful response)
   - Can skip stages or regress based on user responses

3. **Tone Transition Rules**
   - Map classifications to tone changes
   - Example: "expresses confusion" → shift to warmer tone
   - Example: "insightful response" → shift to delighted/impressed tone

---

## PHASE 5: LLM PIPELINE IMPLEMENTATION

### What to Code:

1. **Classifier** (`classifier.py`)
   - Define classification categories:
     - demonstrates_understanding
     - expresses_confusion
     - insightful_response
     - asks_clarifying_question
     - minimal_answer
     - off_topic
   - Create classification prompt (short, focused)
   - Use OpenAI API with structured output (JSON mode)
   - Function: `classify_user_input(user_text) → category`

2. **Response Generator** (`generator.py`)
   - Build comprehensive system prompt from:
     - Character design document
     - Current state (topic, stage, tone)
   - Build user prompt from:
     - RAG passages (retrieved based on topic)
     - Conversation history (last 3-5 exchanges)
     - Current user input
     - Classification result
   - Use OpenAI API to generate response
   - Function: `generate_response(state, user_input, classification, history) → text`

3. **Exception Handling**
   - Detect off-topic/anachronistic inputs via classification
   - Create character-appropriate deflection responses
   - Return conversation to philosophical topics
   - Stay in character (e.g., Socrates: "My friend, I know nothing of this 'internet' you speak of...")

---

## PHASE 6: FRONT-END DEVELOPMENT

### What to Code:

1. **HTML Structure** (`templates/index.html`)
   - Title and philosopher introduction
   - Character image display (changes with tone)
   - Text display area (philosopher's responses)
   - User input field
   - Submit button
   - Optional: progress indicator (topics covered)

2. **CSS Styling** (`static/css/style.css`)
   - Design appropriate theme for your philosopher
   - Responsive layout
   - Tone-based styling changes (colors, fonts)

3. **JavaScript Frontend** (`static/js/app.js`)
   - Handle user input submission
   - Send POST request to Flask backend
   - Receive response and new state
   - Update UI:
     - Display philosopher's text
     - Change image based on tone
     - Update any state indicators
   - Manage conversation history display

4. **Flask Backend** (`app.py`)
   - Route for serving main page
   - API endpoint: `POST /chat`
     - Receives user input
     - Calls classifier
     - Updates state machine
     - Retrieves RAG passages
     - Generates response
     - Returns response + state info
   - Route for serving static assets

---

## PHASE 7: TESTING AND REFINEMENT

### What to Test:

1. **RAG System**
   - Test various queries
   - Verify relevant passages are retrieved
   - Adjust chunk size if needed
   - Test different embedding models if results poor

2. **State Machine**
   - Walk through complete conversation flow
   - Test all topic transitions
   - Verify stage progressions work
   - Test tone transitions

3. **Classifier Accuracy**
   - Test with various user inputs
   - Verify classifications are reasonable
   - Refine classification prompt if needed

4. **Response Quality**
   - Check character consistency
   - Verify RAG context is used appropriately
   - Test exception handling (off-topic inputs)
   - Ensure responses advance conversation

5. **Full Integration**
   - Complete conversations from start to finish
   - Test edge cases
   - Get feedback from others

6. **Write Reflection** (~500 words)
   - What succeeded and failed
   - Strengths and limitations of LLMs observed
   - Challenges encountered
   - Lessons learned

---

## IMPLEMENTATION ORDER (RECOMMENDED)

### Week 1:
- ✅ Choose philosopher
- ✅ Download texts
- ✅ Set up environment
- ✅ Write character design document
- ✅ Design topic graph and tone states

### Week 2:
- ✅ Implement text chunking
- ✅ Set up ChromaDB
- ✅ Build and test RAG system
- ✅ Get OpenAI API key from professor

### Week 3:
- ✅ Implement state machine
- ✅ Implement classifier
- ✅ Implement response generator
- ✅ Test LLM pipeline

### Week 4:
- ✅ Build front-end HTML/CSS/JS
- ✅ Build Flask backend
- ✅ Integration testing
- ✅ Refinement and bug fixes
- ✅ Write reflection
- ✅ Demo and submit

---

## QUICK START CHECKLIST

Before coding anything:
- [ ] Choose your philosopher
- [ ] Download at least 10,000 words of their texts
- [ ] Skim the texts to understand key ideas
- [ ] Identify 4+ major topics/concepts
- [ ] Write character design document
- [ ] Design topic graph
- [ ] Design tone states
- [ ] Set up Python environment
- [ ] Install all dependencies
- [ ] Wait for/receive OpenAI API key
- [ ] Create .env file with key

Then start coding:
- [ ] Text chunker and RAG system first (most foundational)
- [ ] Test RAG with sample queries
- [ ] State machine
- [ ] LLM classifier and generator
- [ ] Front-end last (easier to test backend first with print statements)

---

## KEY DECISIONS TO MAKE

1. **Chunk Size**: Start with 300 words, experiment between 200-500
2. **Number of RAG Passages**: Start with 3, adjust based on prompt length
3. **OpenAI Model**: Use GPT-4 for quality, GPT-3.5-turbo for cost savings
4. **Classification Categories**: Start with the 6 suggested, add more if needed
5. **Number of Topics**: Minimum 4, but 5-6 gives better conversation depth
6. **Visual Style**: AI-generated, hand-drawn, classical art, modern graphics?

---

## COST MANAGEMENT

- RAG encoding is one-time (local, free after setup)
- OpenAI API costs per call:
  - Classification: Small prompt = ~$0.001 per call
  - Generation: Larger prompt = ~$0.01-0.05 per call
- Budget for ~100-200 test conversations
- Use shorter prompts where possible
- Consider GPT-3.5-turbo for development, GPT-4 for final version

---

## COMMON PITFALLS TO AVOID

1. ❌ Don't hardcode the API key in your code
2. ❌ Don't make chunks too small (<100 words) or too large (>1000 words)
3. ❌ Don't forget to include conversation history in generation
4. ❌ Don't use the same LLM call for classification AND generation
5. ❌ Don't make the character break character for off-topic questions
6. ❌ Don't forget to handle empty/very short user inputs
7. ❌ Don't upload your .env file to GitHub!

---

## RESOURCES

- ChromaDB Docs: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- OpenAI API Docs: https://platform.openai.com/docs/
- Project Gutenberg: https://www.gutenberg.org/
- Internet Classics: http://classics.mit.edu/
- RAG Overview: https://python.langchain.com/docs/use_cases/question_answering/
