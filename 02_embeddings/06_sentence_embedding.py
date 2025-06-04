import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

def sentence_embeddings():
    """
    Generates sentence embeddings using a pre-trained SentenceTransformer model.
    Visualizes the embeddings in a 2D scatter plot using PCA.
    Saves the plot as 'sentence_embeddings.png'.
    """
    # Load SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Define sample sentences
    sentences = [
        "Life insurance provides financial security.",
        "Beneficiaries receive claims after policyholder's death.",
        "Premium payments ensure continued coverage.",
        "Investment-linked policies offer financial growth.",
        "Risk assessment affects insurance pricing."
    ]

    # Generate embeddings
    embeddings = model.encode(sentences)

    print("Sentence Embedding shape:", embeddings.shape)
    print("First sentence embedding (truncated):", embeddings[0][:5])

    # Reduce embeddings to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    for i, sentence in enumerate(sentences):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, label=sentence)
        plt.text(x + 0.01, y + 0.01, sentence, fontsize=10)

    plt.title("Sentence Embeddings Visualization (Life Insurance)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save plot as PNG
    plt.savefig("06_sentence_embeddings.png", dpi=300)
    print("Sentence embeddings plot saved as '06_sentence_embeddings.png'.")

# Run function
sentence_embeddings()