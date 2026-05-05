import numpy as np
import matplotlib.pyplot as plt

# Read makemore-generated names
with open('generated_names_makemore.txt', 'r') as f:
    names = f.read().splitlines()

print(f"Loaded {len(names)} makemore-generated names")

# Build transition matrix
# Create a mapping of characters to indices
chars = ['.'] + sorted(list(set(''.join(names))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# Initialize count matrix
n_chars = len(chars)
counts = np.zeros((n_chars, n_chars), dtype=np.int32)

# Count transitions
for name in names:
    name = '.' + name + '.'
    for i in range(len(name) - 1):
        first = char_to_idx[name[i]]
        second = char_to_idx[name[i + 1]]
        counts[first, second] += 1

print(f"Total transitions: {counts.sum()}")

# Convert counts to probabilities
probs = counts.astype(np.float32)
# Normalize each row to sum to 1
probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)  # Add small epsilon to avoid division by zero

# Create heatmap
plt.figure(figsize=(16, 16))
plt.imshow(probs, cmap='hot', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks(range(n_chars), chars)
plt.yticks(range(n_chars), chars)
plt.xlabel('Second Letter')
plt.ylabel('First Letter')
plt.title('Letter Transition Probabilities in Makemore-Generated Names')

plt.tight_layout()
plt.savefig('name_heatmap_makemore.png', dpi=150)
print("Heatmap saved to name_heatmap_makemore.png")

# Compare with real data statistics
print("\n=== COMPARISON WITH REAL DATA ===\n")

# Load the real data probabilities for comparison
real_probs = np.load('transition_probs.npy')
real_char_to_idx = np.load('char_to_idx.npy', allow_pickle=True).item()

# Starting letters comparison
print("Starting letters in makemore-generated names:")
start_row = probs[char_to_idx['.'], :]
start_probs = [(chars[i], start_row[i]) for i in range(1, n_chars)]
start_probs.sort(key=lambda x: x[1], reverse=True)
print("Three most likely starting letters:")
for i in range(min(3, len(start_probs))):
    if start_probs[i][1] > 0:
        print(f"  {start_probs[i][0]}: {start_probs[i][1]:.4f}")

# Ending letters comparison
print("\nEnding letters in makemore-generated names:")
end_col = probs[:, char_to_idx['.']]
end_probs = [(chars[i], end_col[i]) for i in range(1, n_chars)]
end_probs.sort(key=lambda x: x[1], reverse=True)
print("Three most likely ending letters:")
for i in range(min(3, len(end_probs))):
    if end_probs[i][1] > 0:
        print(f"  {end_probs[i][0]}: {end_probs[i][1]:.4f}")

# Sample some names to show
print("\nSample of makemore-generated names:")
for i in range(min(20, len(names))):
    print(f"  {names[i].capitalize()}")
