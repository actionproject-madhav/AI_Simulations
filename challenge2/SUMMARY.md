# Name Generation Project - Summary

## Overview
This project explores character-level language modeling for name generation using both statistical (bigram) models and neural network (transformer) models.

## Part 1: Statistical Analysis of Real Names

### Dataset
- **Source**: `names.txt` from makemore repository
- **Total names**: 32,033
- **Total character transitions**: 228,146

### Statistical Questions Answered

#### 1. Starting Letters
**Three most likely starting letters:**
- `a`: 0.1377 (13.77%)
- `k`: 0.0925 (9.25%)
- `m`: 0.0792 (7.92%)

**Three least likely starting letters:**
- `x`: 0.0042 (0.42%)
- `q`: 0.0029 (0.29%)
- `u`: 0.0024 (0.24%)

**Analysis**: Names tend to start with vowels (especially 'a') and common consonants. Rare letters like 'x', 'q', and 'u' are very uncommon as starting letters.

#### 2. Ending Letters
**Three most likely ending letters:**
- `n`: 0.3690 (36.90%)
- `h`: 0.3163 (31.63%)
- `x`: 0.2353 (23.53%)

**Three least likely ending letters:**
- `p`: 0.0322 (3.22%)
- `c`: 0.0275 (2.75%)
- `j`: 0.0245 (2.45%)

**Analysis**: Names show strong preferences for ending with 'n', 'h', or 'x'. The high probability for 'n' likely reflects common name endings like "-an", "-en", "-on", "-yn". The letter 'h' appears in endings like "-ah", "-eh", and "-leigh".

#### 3. Letters Following 'q'
**Letters that follow 'q' (other than 'u'):**
- `.` (end of name): 0.1029
- `a`: 0.0478
- `i`: 0.0478
- `w`: 0.0110
- `m`: 0.0074
- `o`: 0.0074
- `s`: 0.0074
- `e`: 0.0037
- `l`: 0.0037
- `r`: 0.0037

**Analysis**: While 'u' is by far the most common letter following 'q' (over 76%), there are names where 'q' is followed by other letters or ends the name. This includes names from various cultural backgrounds.

#### 4. Names Starting with 'x'
**Most likely second letter for names starting with 'x':**
- `.` (end of name): 0.2353 (23.53%)

**Analysis**: Surprisingly, the most likely "second letter" for names starting with 'x' is the end-of-name marker, meaning many single-letter names or very short names start with 'x'. After that, names like "Xavier", "Ximena", "Xander" contribute to the patterns.

### Heatmap Analysis (Real Names)
The transition matrix heatmap (`name_heatmap_real.png`) shows:
- **Strong diagonal patterns**: Letters often repeat (e.g., 'll', 'nn', 'ss')
- **Bright 'q→u' cell**: Nearly 80% probability, showing the strong English language rule
- **Vowel-consonant patterns**: Bright cells showing consonants frequently followed by vowels
- **Common digraphs**: Visible patterns for 'th', 'ch', 'sh', 'an', 'en', 'ly'

## Part 2: Statistical Model Generation

### Method
Used a bigram model that:
1. Built transition probability matrix from training data
2. Generated names by sampling from character distributions
3. Started from '.' and sampled until reaching '.' again
4. Rejected names shorter than 3 characters

### Generated Names (25 samples)
1. Zeshanay
2. Siesan
3. Lolosucavabacevotaia
4. Axzydalionahulel
5. Devidhngrion
6. Ketan
7. Lelelaynotia
8. Aiameliay
9. Khaeeek
10. Kyaheon
11. Madie
12. Omilendekaerteyariey
13. Azieasaharilemeyn
14. Cedennn
15. Ddaeana
16. Ssh
17. Aynanavaury
18. Gshan
19. Taseinz
20. Don
21. Liane
22. Dre
23. Yailakawa
24. Kan
25. Kek

### Observations on Statistical Model
**Realistic names:**
- Ketan, Madie, Liane, Don - these sound like plausible real names

**Characteristics:**
- Some names are extremely long (e.g., "Lolosucavabacevotaia")
- Repetitive patterns appear (e.g., "Lelelaynotia", "Cedennn")
- Some contain unusual consonant clusters
- The model captures local patterns but lacks global coherence

**Limitations:**
- No understanding of name length distributions
- Can create very long sequences due to sampling randomness
- No semantic understanding of what makes a "good" name
- Only considers immediate previous character (bigram model)

## Part 3: Neural Network Training

### Model Architecture
- **Type**: Transformer (similar to GPT-2)
- **Parameters**: 204,544 (~0.20M)
- **Layers**: 4
- **Attention heads**: 4
- **Embedding dimension**: 64
- **Context window**: 16 characters

### Training Details
- **Training steps**: 100,000
- **Training set**: 31,033 names
- **Test set**: 1,000 names
- **Initial loss**: ~3.3
- **Final train loss**: ~1.56
- **Final test loss**: ~2.11
- **Training time**: ~20 minutes on CPU

### Loss Progression
The model showed steady improvement:
- Step 10,000: Loss ~1.85
- Step 20,000: Loss ~1.70
- Step 40,000: Loss ~1.63
- Step 60,000: Loss ~1.60
- Step 80,000: Loss ~1.58
- Step 100,000: Loss ~1.56

## Part 4: Makemore-Generated Names

### Generated Dataset
- **Total names generated**: 635 (all unique and new)
- **Names submitted**: 250
- **File**: `generated_names_makemore.txt`

### Sample Names
Aadee, Aadrianna, Aalena, Abdamadad, Abdimus, Abdulkasim, Abrottan, Abubakum, Adanson, Addix, Adesa, Adifah, Adnae, Adolwamiz, Aegan, Ahla, Ahlian, Aiha, Ainla, Ainus, Ajadh, Ajaiyon, Akaro, Akazia, Akio, Alaina, Alamiah, Alanara, Alandra, Aleah...

### Observations on Neural Network Names
**Quality:**
- Most names are pronounceable and look realistic
- Better length control than the statistical model
- More diverse character combinations
- Captures longer-range dependencies

**Characteristics:**
- Maintains phonological patterns from training data
- Creates novel combinations that sound plausible
- Some unusual but interesting combinations (e.g., "Abdulkasim", "Adolwamiz")
- Generally between 4-10 characters in length

## Part 5: Comparison of Heatmaps

### Real Names Heatmap
**Key patterns:**
- Strong 'q→u' pattern (white/bright yellow)
- Clear vowel-after-consonant patterns
- Common ending patterns visible in '→.' column
- Balanced distribution across common letters

### Makemore-Generated Names Heatmap
**Key patterns:**
- Similar overall structure to real names
- 'q→u' pattern present but less pronounced (fewer 'q' instances)
- Stronger starting letter bias (more concentrated '.' row)
- Ending patterns similar to real data

### Similarities
1. **Letter transition patterns**: Both show similar consonant-vowel alternation
2. **Common digraphs**: 'th', 'ch', 'sh' patterns appear in both
3. **Ending preferences**: Both favor 'n', 'h', and 'x' as ending letters
4. **Vowel usage**: Similar vowel distribution patterns

### Differences
1. **Starting letter distribution**: Makemore shows stronger bias toward 'a' (37.6% vs 13.77%)
2. **Diversity**: Real names show more uniform distribution across letters
3. **Rare combinations**: Some cells in makemore heatmap are empty due to smaller sample size (250 vs 32,033)
4. **Pattern smoothness**: Neural network creates slightly smoother transition patterns

### Statistical Comparison

**Starting letters:**
- Real data: a (13.77%), k (9.25%), m (7.92%)
- Makemore: a (37.60%), d (20.00%), e (9.60%)

**Ending letters:**
- Real data: n (36.90%), h (31.63%), x (23.53%)
- Makemore: n (41.27%), x (40.00%), h (32.73%)

**Analysis**: The neural network learned the general structure but shows some biases, particularly in starting letters. The ending letter distribution is remarkably similar, suggesting the model successfully learned this important pattern.

## Conclusions

### Statistical (Bigram) Model
**Strengths:**
- Simple to implement and understand
- Fast to train and generate
- Captures local character patterns

**Weaknesses:**
- No long-range dependencies
- Can generate very long or nonsensical names
- No understanding of name structure

### Neural Network (Transformer) Model
**Strengths:**
- Learns longer-range dependencies
- Better length control
- More realistic outputs
- Captures complex patterns

**Weaknesses:**
- Requires more training time and data
- More complex to implement
- Can overfit to training data patterns

### Overall Findings
The transformer model significantly outperforms the simple bigram model in generating realistic names. It successfully learns:
1. Appropriate name lengths
2. Phonological patterns
3. Letter combination rules
4. Structural patterns from the training data

The heatmap comparison reveals that while the neural network captures the essential statistical properties of names, it also shows some biases due to the learning process and potentially the smaller sample size in the generated set.

## Files Included
1. `name_heatmap_real.png` - Heatmap of real names
2. `name_heatmap_makemore.png` - Heatmap of makemore-generated names
3. `generated_names_statistical.txt` - 25 names from bigram model
4. `generated_names_makemore.txt` - 250 names from neural network
5. `name_stats.py` - Statistical analysis script for real names
6. `makemore_stats.py` - Statistical analysis script for generated names
7. `generate_names.py` - Bigram model generation script
