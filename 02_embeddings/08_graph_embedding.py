import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Graph Embeddings
def graph_embeddings():
    """
    Generates graph embeddings using Node2Vec, reduces dimensionality,
    and visualizes the embeddings in 2D using PCA and t-SNE.
    """
    # Create the Karate Club graph (a well-known social network dataset)

    # Node2Vec is an algorithmic framework for learning continuous feature representations for nodes in networks
    G = nx.karate_club_graph()

    # Train Node2Vec model
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

    # Parameters:
    # - dimensions=64: Dimensionality of the node embeddings
    # - walk_length=30: Length of walk per source
    # - num_walks=200: Number of walks per source
    # - workers=4: Number of CPU cores to use
    
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # Additional parameters for fit:
    # - window=10: Maximum distance between the current and predicted node in the random walk
    # - min_count=1: Ignores all nodes with total frequency lower than this
    # - batch_words=4: Number of words to be processed in a single batch
    
    print("Graph Embedding for node 0:", model.wv['0'])

    # Extract embeddings for all nodes
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    
    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(node_embeddings)

    # Reduce dimensions using t-SNE for better clustering visualization
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    tsne_embeddings = tsne.fit_transform(node_embeddings)

    # Plot PCA-based visualization
    plt.figure(figsize=(8, 6))
    for i, node in enumerate(G.nodes()):
        x, y = pca_embeddings[i]
        plt.scatter(x, y, label=str(node))
        plt.text(x + 0.01, y + 0.01, str(node), fontsize=10)

    plt.title("Graph Embeddings Visualization (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("graph_embeddings_pca.png", dpi=300)
    print("Graph embeddings plot saved as 'graph_embeddings_pca.png'.")

    # Plot t-SNE-based visualization
    plt.figure(figsize=(8, 6))
    for i, node in enumerate(G.nodes()):
        x, y = tsne_embeddings[i]
        plt.scatter(x, y, label=str(node))
        plt.text(x + 0.01, y + 0.01, str(node), fontsize=10)

    plt.title("Graph Embeddings Visualization (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig("08_graph_embeddings_tsne.png", dpi=300)
    print("Graph embeddings plot saved as '08_graph_embeddings_tsne.png'.")

# Run function
graph_embeddings()
