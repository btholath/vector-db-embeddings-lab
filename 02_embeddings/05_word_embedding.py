import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Word Embeddings (Word2Vec)
from gensim.models import Word2Vec

def word_embeddings_1():
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=1)
    print("\nWord Embedding for 'cat':\n", model.wv['cat'])
    print("Similarity between 'cat' and 'dog':", model.wv.similarity('cat', 'dog'))
    # Extract word vectors
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])

    # Reduce dimensions to 2D for plotting
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plot the word embeddings
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced_vectors[i]
        plt.scatter(x, y, label=word)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)

    plt.title("Word Embeddings Visualization (Animals & Sounds)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save plot as PNG
    plt.savefig("05_word_embeddings_animals.png", dpi=300)
    print("Word embeddings plot saved as '05_word_embeddings_animals.png'.")


def word_embeddings_2():
    """
    Uses life insurance-related words in training sentences
    Builds word embeddings to find semantic relationships between insurance terms
    Computes similarity between "policy" and "coverage"
    Outputs the numerical vector representation of "insurance"
    """
    sentences = [
        ['life', 'insurance', 'policy'],
        ['premium', 'coverage', 'beneficiary'],
        ['claim', 'approval', 'risk'],
        ['term', 'whole', 'insurance'],
        ['investment', 'returns', 'security']
    ]
    model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=1)
    print("\nWord Embedding for 'insurance':\n", model.wv['insurance'])
    print("Similarity between 'policy' and 'coverage':", model.wv.similarity('policy', 'coverage'))

    # create scatter plot
    # Extract word vectors
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])

    # Reduce dimensions to 2D using PCA for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plot embeddings
    plt.figure(figsize=(8, 6))
    for i, word in enumerate(words):
        x, y = reduced_vectors[i]
        plt.scatter(x, y, label=word)
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)

    plt.title("Word Embeddings Visualization (Life Insurance Terms)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save plot as PNG
    plt.savefig("05_word_embeddings_2.png", dpi=300)
    print("Word embeddings plot saved as '05_word_embeddings_2.png'.")


if __name__ == "__main__":
    word_embeddings_1()
    word_embeddings_2()

