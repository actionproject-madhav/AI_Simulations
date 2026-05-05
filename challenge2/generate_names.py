import numpy as np

# Load the transition probabilities and character mapping
probs = np.load('transition_probs.npy')
char_to_idx = np.load('char_to_idx.npy', allow_pickle=True).item()
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

def random_choice(distribution):
    """Choose the next letter based on the probability distribution."""
    return np.random.choice(len(distribution), p=distribution)

def generate_name(min_length=3):
    """Generate a single name using the transition probabilities."""
    letter_idx = char_to_idx['.']
    name = ''

    while True:
        # Choose the next letter from the distribution
        next_idx = random_choice(probs[letter_idx])
        next_letter = idx_to_char[next_idx]

        if next_letter == '.':
            # End of name
            if len(name) >= min_length:
                break
            else:
                # Name too short, restart
                letter_idx = char_to_idx['.']
                name = ''
        else:
            name += next_letter
            letter_idx = next_idx

    return name

# Generate 25 names
print("Generating 25 names using statistical model:\n")
names = []
for i in range(25):
    name = generate_name(min_length=3)
    names.append(name)
    print(f"{i+1:2d}. {name.capitalize()}")

# Save the generated names
with open('generated_names_statistical.txt', 'w') as f:
    for name in names:
        f.write(name + '\n')

print("\nGenerated names saved to generated_names_statistical.txt")
