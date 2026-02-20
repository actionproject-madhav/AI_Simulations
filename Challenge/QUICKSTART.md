# Quick Start Guide - Buddha AI

Get the system running in 5 minutes.

## Step 1: Verify Setup

Make sure you have:
- [x] Python 3.11+ installed
- [x] Node.js 16+ installed
- [x] OpenAI API key from professor
- [x] Text files in `texts/` folder (already done)
- [x] Images in `static/images/` folder (already done)

## Step 2: Configure Environment

1. **Check your .env file:**
```bash
cat .env
```

Should look like:
```
OPENAI_API_KEY=sk-proj-...your-actual-key...
```

If it's empty or wrong format, fix it:
```bash
# Open in editor
nano .env

# Add (no quotes, no spaces around =):
OPENAI_API_KEY=your_key_here

# Save and exit (Ctrl+X, then Y, then Enter)
```

2. **Verify configuration:**
```bash
source venv/bin/activate
python config.py
```

Should say `Valid: True`. If not, check your .env file format.

## Step 3: Run Backend

Open a terminal window:

```bash
cd /Users/madhav/Documents/AI_Simulations/Challenge

# Activate Python environment
source venv/bin/activate

# Start Flask server
python app.py
```

You should see:
```
============================================================
Buddha AI - Starting Server
============================================================
Initializing RAG system...
Loaded existing collection with 462 chunks
Initializing classifier...
Initializing generator...

Server running on http://localhost:5000
============================================================
```

Leave this terminal running.

## Step 4: Run Frontend

Open a NEW terminal window:

```bash
cd /Users/madhav/Documents/AI_Simulations/Challenge/frontend

# Install dependencies (first time only)
npm install

# Start React dev server
npm start
```

Your browser should automatically open to `http://localhost:3000`.

If not, manually visit: http://localhost:3000

## Step 5: Test the System

1. **Initial Greeting**
   - When page loads, Buddha should greet you
   - You'll see: "Welcome, friend. Let us begin with the foundation..."

2. **Try Different Responses**
   - Confused: "I don't understand what you mean"
     → Buddha should become compassionate (warm tone)

   - Engaged: "So attachment to things causes suffering?"
     → Buddha should become teaching (probing tone)

   - Insightful: "Oh! Everything changes, so clinging to anything is futile!"
     → Buddha should become meditative (contemplative tone)

   - Minimal: "yeah" or "ok"
     → Buddha should become challenging (stern tone)

3. **Watch for State Changes**
   - Left sidebar shows current topic, stage, and tone
   - Buddha's image changes with tone
   - As you progress, topic advances: Four Noble Truths → Middle Way → Impermanence → Non-Self

## Common Issues

### Backend won't start

**Error: "OPENAI_API_KEY not found"**
```bash
# Fix .env file format
echo "OPENAI_API_KEY=sk-proj-your-key" > .env
python config.py  # Should say Valid: True
```

**Error: "ModuleNotFoundError"**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend won't start

**Error: "Cannot find module"**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

**Port 3000 already in use**
```bash
# Kill existing process
lsof -ti:3000 | xargs kill -9
npm start
```

### Connection refused / CORS errors

Make sure:
1. Backend is running on port 5000
2. Frontend is running on port 3000
3. Check `frontend/package.json` has `"proxy": "http://localhost:5000"`

### Slow responses

First message takes longer (initializing models). Subsequent messages should be 3-5 seconds.

If consistently slow:
- Check internet connection
- Try switching to gpt-3.5-turbo in .env:
  ```
  OPENAI_MODEL_GENERATOR=gpt-3.5-turbo
  ```

## Testing Checklist

Try this conversation to test all features:

1. Start: Buddha greets you
2. You: "What is suffering?"
   - Should get compassionate/teaching response with examples
3. You: "I don't really get it"
   - Should become more compassionate, gentler explanation
4. You: "Oh, so suffering is more than just physical pain?"
   - Should confirm and probe deeper
5. You: "It's everything we cling to that causes pain"
   - Should become meditative, acknowledge insight
6. You: "yeah"
   - Should become challenging, demand better engagement
7. You: "What about Bitcoin?"
   - Should redirect to philosophy, stay in character
8. Continue until topic changes to Middle Way
   - Should introduce new topic smoothly

If all the above work, your system is functioning correctly!

## Production Build (Optional)

To create a single deployable app:

```bash
# Build React frontend
cd frontend
npm run build
cd ..

# Run Flask (serves both API and frontend)
source venv/bin/activate
python app.py
```

Visit: http://localhost:5000 (single server for everything)

## Need Help?

1. Check `README.md` for detailed documentation
2. Test individual components:
   ```bash
   python backend/rag_system.py
   python backend/state_machine.py
   ```
3. Check Flask logs in terminal for error messages
4. Check browser console (F12) for frontend errors

## You're Ready!

The system is now running. Have a philosophical dialogue with Buddha and observe how the conversation adapts to your engagement level.

Enjoy exploring Buddhist philosophy!
