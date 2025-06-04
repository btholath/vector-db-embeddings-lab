import numpy as np
import matplotlib.pyplot as plt

# Simple 2D embeddings for weather and seasons words
word_embeddings = {
    "summer": [0.9, 0.9],
    "winter": [0.1, 0.95],
    "spring": [0.8, 0.7],
    "autumn": [0.5, 0.8],
    "rain": [0.3, 0.6],
    "snow": [0.2, 0.95],
    "sun": [0.95, 0.85],
    "cloud": [0.6, 0.6],
    "storm": [0.4, 0.3],
    "wind": [0.7, 0.4]
}

# Plotting the embeddings
plt.figure(figsize=(10, 8))
for word, embedding in word_embeddings.items():
    plt.scatter(embedding[0], embedding[1], label=word)
    plt.annotate(word, (embedding[0]+0.01, embedding[1]+0.01))

plt.title("Simple 2D Word Embeddings (Seasons & Weather)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Print out the embeddings
for word, embedding in word_embeddings.items():
    print(f"{word}: {embedding}")
