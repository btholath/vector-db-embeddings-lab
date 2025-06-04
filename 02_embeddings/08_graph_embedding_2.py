# 08_graph_embedding_2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA

"""
graph embeddings for insurance fraud detection using Node2Vec and NetworkX. This script creates a fraud detection network, where nodes represent policyholders, claims, adjusters, and edges denote relationships (e.g., shared addresses, previous claims, or suspicious connections).
Constructs a real-world fraud detection network, where:
- Policyholders, adjusters, and claims are nodes.
- Edges represent suspicious relationships (e.g., shared addresses, previous claims).
    Applies Node2Vec to learn graph embeddings.
    Uses PCA for dimensionality reduction and visualization.
    Saves the fraud risk network as a PNG file (insurance_fraud_graph_pca.png).
"""
def insurance_fraud_graph():
    """
    Creates an insurance fraud detection graph, applies Node2Vec embeddings,
    reduces dimensionality using PCA, and visualizes the fraud risk network.
    """
    # Create a fraud detection graph
    G = nx.Graph()

    # Add nodes representing policyholders, claims, and adjusters
    policyholders = ["Alice", "Bob", "Charlie", "David"]
    adjusters = ["Adjuster1", "Adjuster2"]
    claims = ["ClaimA", "ClaimB", "ClaimC"]

    # Add edges based on relationships (e.g., common addresses, shared adjusters)
    edges = [
        ("Alice", "ClaimA"), ("Bob", "ClaimB"), ("Charlie", "ClaimC"),
        ("ClaimA", "Adjuster1"), ("ClaimB", "Adjuster1"), ("ClaimC", "Adjuster2"),
        ("Alice", "Bob"),  # Shared address
        ("Bob", "Charlie"),  # Frequent previous claims together
        ("Charlie", "David"),  # Same employer
        ("ClaimA", "ClaimB")  # Suspiciously linked claims
    ]
    
    G.add_nodes_from(policyholders, type="policyholder")
    G.add_nodes_from(adjusters, type="adjuster")
    G.add_nodes_from(claims, type="claim")
    G.add_edges_from(edges)

    # Train Node2Vec model to extract embeddings
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])

    # Reduce dimensions to 2D using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(node_embeddings)

    # Visualize the fraud risk graph using PCA embeddings
    plt.figure(figsize=(8, 6))
    for i, node in enumerate(G.nodes()):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, label=str(node))
        plt.text(x + 0.01, y + 0.01, str(node), fontsize=10)

    plt.title("Insurance Fraud Detection Graph (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("08_insurance_fraud_graph_pca.png", dpi=300)
    print("Insurance fraud detection graph saved as '08_insurance_fraud_graph_pca.png'.")

# Run function
insurance_fraud_graph()