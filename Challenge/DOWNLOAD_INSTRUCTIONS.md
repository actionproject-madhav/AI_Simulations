# Download Instructions for Buddha AI Project

## STEP 1: Download Texts (to Challenge/texts/)

### Primary Text: The Dhammapada
1. Go to: https://www.gutenberg.org/ebooks/2017
2. Click "Plain Text UTF-8" download
3. Save as: `texts/dhammapada.txt`
4. Size: ~15,000 words ✓

### Secondary Texts (choose one or more to reach 10,000+ words total):

**Option A: Buddhism in Translations** by Henry Clarke Warren
- URL: https://www.sacred-texts.com/bud/bits/index.htm
- Download the full text
- Save as: `texts/buddhism_in_translations.txt`

**Option B: The Udana**
- URL: https://www.sacred-texts.com/bud/udana.htm
- Click download link or copy text
- Save as: `texts/udana.txt`

**Option C: Selected Suttas from Access to Insight**
- URL: https://accesstoinsight.org/
- Browse and download major suttas (Metta Sutta, Heart Sutra, etc.)
- Combine into: `texts/selected_suttas.txt`

---

## STEP 2: Generate/Download Images (to Challenge/static/images/)

You need **at least 4 images** representing different emotional tones.

### Recommended Tones for Buddha:

1. **compassionate.png** - Warm, gentle smile, welcoming
2. **meditative.png** - Serene, peaceful, eyes closed/half-closed
3. **teaching.png** - Engaged, hand gestures, focused
4. **challenging.png** - Serious, testing, stern but not angry

### Where to Get Images:

#### Option A: AI Generation (Recommended - Consistent Style)
Use ChatGPT, DALL-E, Midjourney, or similar:

**Prompt Templates:**
```
"Buddha in meditation pose, compassionate expression, warm golden lighting,
digital art style, peaceful atmosphere, front-facing portrait"

"Buddha teaching, hand in dharma wheel mudra, focused expression,
serene background, digital art, traditional yet modern"

"Buddha in deep meditation, eyes closed, completely peaceful,
lotus position, soft blue lighting, spiritual digital art"

"Buddha with serious expression, testing a student, wise and stern,
traditional robes, contemplative atmosphere, digital art"
```

**Visual Novel Style (like Kant-senpai):**
```
"Buddha as anime character, compassionate expression, traditional robes,
soft lighting, visual novel style, detailed portrait"
```

#### Option B: Public Domain Classical Art
- Wikimedia Commons: https://commons.wikimedia.org/wiki/Category:Gautama_Buddha
- Search for different Buddha poses/expressions
- Download high-quality versions
- Resize to consistent dimensions (e.g., 512x512 or 1024x1024)

#### Option C: Create Your Own
- Digital art
- Photography with editing
- Collage/mixed media

### Image Specifications:
- Format: PNG or JPG
- Recommended size: 512x512 or 1024x1024 pixels
- Consistent style across all images
- Clear, front-facing portraits work best
- Name them clearly: `compassionate.png`, `meditative.png`, etc.

---

## STEP 3: Verify Downloads

After downloading, your Challenge/texts/ folder should contain:
```
texts/
├── dhammapada.txt
└── [one or more additional texts totaling 10,000+ words]
```

Your Challenge/static/images/ folder should contain:
```
static/images/
├── compassionate.png
├── meditative.png
├── teaching.png
└── challenging.png
```

Run this command to check:
```bash
ls -lh texts/
ls -lh static/images/
wc -w texts/*.txt  # Should show 10,000+ total words
```

---

## Quick Links Summary

**Texts:**
- Dhammapada: https://www.gutenberg.org/ebooks/2017
- Buddhism in Translations: https://www.sacred-texts.com/bud/bits/index.htm
- Udana: https://www.sacred-texts.com/bud/udana.htm
- Access to Insight: https://accesstoinsight.org/

**Images:**
- Use ChatGPT/DALL-E for AI generation
- Or Wikimedia Commons for public domain art

---

## Next Steps (After Downloads Complete)

1. ✓ Texts downloaded to `texts/`
2. ✓ Images saved to `static/images/`
3. → Write character design document
4. → Design topic graph
5. → Design tone states
6. → Set up Python environment
7. → Start coding!
