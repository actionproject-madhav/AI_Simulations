import numpy as np
import matplotlib.pyplot as plt

# Read names from file
with open('makemore/names.txt', 'r') as f:
    names = f.read().splitlines()

print(f"Loaded {len(names)} names")

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
probs = probs / probs.sum(axis=1, keepdims=True)

# Create heatmap
plt.figure(figsize=(16, 16))
plt.imshow(probs, cmap='hot', interpolation='nearest')
plt.colorbar(label='Probability')
plt.xticks(range(n_chars), chars)
plt.yticks(range(n_chars), chars)
plt.xlabel('Second Letter')
plt.ylabel('First Letter')
plt.title('Letter Transition Probabilities in Real Names')

# Add text annotations for better readability (optional, can be slow for large matrices)
# Uncomment if you want to see exact values
# for i in range(n_chars):
#     for j in range(n_chars):
#         if probs[i, j] > 0.01:  # Only show significant probabilities
#             plt.text(j, i, f'{probs[i, j]:.2f}', ha='center', va='center', color='white', fontsize=6)

plt.tight_layout()
plt.savefig('name_heatmap_real.png', dpi=150)
print("Heatmap saved to name_heatmap_real.png")

# Answer questions
print("\n=== STATISTICAL ANALYSIS ===\n")

# Question 1: Most and least likely starting letters
start_row = probs[char_to_idx['.'], :]
start_probs = [(chars[i], start_row[i]) for i in range(1, n_chars)]  # Skip '.' as second letter
start_probs.sort(key=lambda x: x[1], reverse=True)

print("Question 1: Starting letters")
print("Three most likely starting letters:")
for i in range(3):
    print(f"  {start_probs[i][0]}: {start_probs[i][1]:.4f}")
print("Three least likely starting letters:")
for i in range(-3, 0):
    print(f"  {start_probs[i][0]}: {start_probs[i][1]:.4f}")

# Question 2: Ending letters
end_col = probs[:, char_to_idx['.']]
end_probs = [(chars[i], end_col[i]) for i in range(1, n_chars)]  # Skip '.' as first letter
end_probs.sort(key=lambda x: x[1], reverse=True)

print("\nQuestion 2: Ending letters")
print("Three most likely ending letters:")
for i in range(3):
    print(f"  {end_probs[i][0]}: {end_probs[i][1]:.4f}")
print("Three least likely ending letters:")
for i in range(-3, 0):
    print(f"  {end_probs[i][0]}: {end_probs[i][1]:.4f}")

# Question 3: Letters following 'q'
if 'q' in char_to_idx:
    q_row = probs[char_to_idx['q'], :]
    q_followers = [(chars[i], q_row[i]) for i in range(n_chars) if q_row[i] > 0 and chars[i] != 'u']
    print("\nQuestion 3: Letters following 'q' (other than 'u'):")
    if q_followers:
        for ch, prob in q_followers:
            print(f"  {ch}: {prob:.4f}")
    else:
        print("  None - only 'u' follows 'q'")

# Question 4: Most likely second letter for names starting with 'x'
if 'x' in char_to_idx:
    x_row = probs[char_to_idx['x'], :]
    x_followers = [(chars[i], x_row[i]) for i in range(n_chars) if x_row[i] > 0]
    x_followers.sort(key=lambda x: x[1], reverse=True)
    print("\nQuestion 4: Most likely second letter for names starting with 'x':")
    if x_followers:
        print(f"  {x_followers[0][0]}: {x_followers[0][1]:.4f}")
    else:
        print("  No names start with 'x'")

# Save the probability matrix for later use in generation
np.save('transition_probs.npy', probs)
np.save('char_to_idx.npy', char_to_idx)
print("\nTransition probabilities saved to transition_probs.npy")
