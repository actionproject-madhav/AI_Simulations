# RUN THIS FIRST - Simple 3-Step Checklist

## Step 1: Fix Your .env File (2 minutes)

Check current .env:
```bash
cat .env
```

Should look EXACTLY like this:
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

NO QUOTES. NO SPACES around the =

If wrong, fix it:
```bash
nano .env
# Type your key, save with Ctrl+X, Y, Enter
```

Verify it works:
```bash
source venv/bin/activate
python config.py
```

Must say: `Valid: True`

If not valid, .env format is wrong. Fix it.

---

## Step 2: Install Frontend (2 minutes)

```bash
cd frontend
npm install
cd ..
```

Wait for it to finish.

---

## Step 3: Run the System (1 minute)

**Terminal 1 (Backend):**
```bash
source venv/bin/activate
python app.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm start
```

Browser opens to http://localhost:3000

**Talk to Buddha!**

---

## If Something Fails

Read: `QUICKSTART.md` for troubleshooting

Most common issue: .env file format is wrong

Fix:
```bash
# Make sure it's EXACTLY:
OPENAI_API_KEY=your_key_here

# NO spaces around =
# NO quotes
# NO extra lines
```

Then try again.

---

## You're Done!

For full documentation, read:
- `README.md` - Complete documentation
- `PROJECT_COMPLETE.md` - What's been built
- `QUICKSTART.md` - Detailed setup guide
