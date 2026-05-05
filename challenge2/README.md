# Challenge 2: Name Generation - Submission Files

## Assignment Deliverables

### 1. Heatmaps
- **`name_heatmap_real.png`** - Letter transition probability heatmap for the original names dataset (32,033 names)
- **`name_heatmap_makemore.png`** - Letter transition probability heatmap for makemore-generated names (250 names)

### 2. Generated Names
- **`generated_names_statistical.txt`** - 25 names generated using the statistical bigram model
- **`generated_names_makemore.txt`** - 250 names generated using the trained transformer model

### 3. Written Answers
- **`SUMMARY.md`** - Comprehensive document containing:
  - Answers to all statistical questions (starting letters, ending letters, letters after 'q', names starting with 'x')
  - Analysis of the statistical model and its generated names
  - Training details for the neural network
  - Comparison of the two heatmaps with observations about similarities and differences
  - Overall conclusions about both approaches

## Additional Files (Scripts and Data)

### Analysis Scripts
- **`name_stats.py`** - Analyzes real names, builds transition matrix, creates heatmap, answers statistical questions
- **`makemore_stats.py`** - Analyzes makemore-generated names, creates heatmap for comparison
- **`generate_names.py`** - Implements the statistical bigram model to generate names

### Data Files
- **`transition_probs.npy`** - Saved transition probability matrix from real names
- **`char_to_idx.npy`** - Character to index mapping for the transition matrix
- **`makemore/`** - Cloned makemore repository containing:
  - `makemore.py` - Main training and generation script
  - `names.txt` - Original dataset of 32,033 names
- **`names/`** - Training output directory containing:
  - `model.pt` - Trained transformer model (100,000 steps)
  - `events.out.tfevents.*` - TensorBoard training logs

## Quick Summary of Results

### Statistical Analysis (Real Names)
- Most likely starting letters: a (13.77%), k (9.25%), m (7.92%)
- Most likely ending letters: n (36.90%), h (31.63%), x (23.53%)
- Letters after 'q' (besides 'u'): Various letters found, including end-of-name
- Names starting with 'x': Most commonly followed by end-of-name marker (23.53%)

### Model Comparison
- **Bigram Model**: Simple, fast, but generates inconsistent quality names
- **Transformer Model**: Better quality, more realistic, learned complex patterns
- **Heatmap Comparison**: Neural network successfully learned similar transition patterns to real data, with some bias toward certain starting letters

## How to Reproduce

### Generate Statistical Heatmap
```bash
python3 name_stats.py
```

### Generate Names with Bigram Model
```bash
python3 generate_names.py
```

### Train Makemore Model
```bash
python3 makemore/makemore.py -i makemore/names.txt -o names --max-steps 100000
```

### Generate Names with Makemore
```bash
python3 makemore/makemore.py -i makemore/names.txt -o names --sample-only --seed 1
```

### Analyze Makemore-Generated Names
```bash
python3 makemore_stats.py
```

## Project Structure
```
challenge2/
├── README.md                          (this file)
├── SUMMARY.md                         (comprehensive written answers)
├── name_heatmap_real.png             (heatmap #1 - required)
├── name_heatmap_makemore.png         (heatmap #2 - required)
├── generated_names_statistical.txt   (25 names - for reference)
├── generated_names_makemore.txt      (250 names - required)
├── name_stats.py                     (analysis script)
├── makemore_stats.py                 (comparison script)
├── generate_names.py                 (bigram generator)
├── transition_probs.npy              (saved probabilities)
├── char_to_idx.npy                   (character mapping)
├── makemore/                         (cloned repository)
│   ├── makemore.py
│   └── names.txt
└── names/                            (training output)
    ├── model.pt
    └── events.out.tfevents.*
```
