from collections import Counter

# training data
spam_messages = [
    "watch free anime downloads",
    "sell your house now"
]

not_spam_messages = [
    "see you at the house",
    "do you want takeout"
]

# remove common words
stop_words = {"at", "the"}

def tokenize(msg):
    return [w for w in msg.lower().split() if w not in stop_words]

# build word counts
spam_words = []
for msg in spam_messages:
    spam_words.extend(tokenize(msg))

not_spam_words = []
for msg in not_spam_messages:
    not_spam_words.extend(tokenize(msg))

spam_counts = Counter(spam_words)
not_spam_counts = Counter(not_spam_words)

# get vocabulary size
all_words = set(spam_words + not_spam_words)
vocab_size = len(all_words)

print(f"Spam words: {spam_words}")
print(f"Not spam words: {not_spam_words}")
print(f"Vocabulary size: {vocab_size}\n")

def word_prob(word, word_counts, total_words):
    count = word_counts.get(word, 0)
    return (count + 1) / (total_words + vocab_size)

def classify(message):
    words = tokenize(message)

    # calculate likelihoods (ignoring priors since they're equal)
    spam_prob = 1.0
    not_spam_prob = 1.0

    print(f"\nMessage: \"{message}\"")
    print(f"Words: {words}\n")
    print(f"{'Word':<15} {'P(w|spam)':<15} {'P(w|not spam)'}")
    print("-" * 45)

    for word in words:
        p_spam = word_prob(word, spam_counts, len(spam_words))
        p_not_spam = word_prob(word, not_spam_counts, len(not_spam_words))

        print(f"{word:<15} {p_spam:<15.4f} {p_not_spam:<15.4f}")

        spam_prob *= p_spam
        not_spam_prob *= p_not_spam

    print(f"\nP(spam | msg) = {spam_prob:.4e}")
    print(f"P(not spam | msg) = {not_spam_prob:.4e}")

    if spam_prob > not_spam_prob:
        print("Classification: SPAM")
    else:
        print("Classification: NOT SPAM")

    print("=" * 50)

# test the 3 practice problems
classify("watch anime now")
classify("takeout and anime at my house")
classify("sell me your anime collection")
