import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def insurance_fraud_clustering():
    """
    Creates an insurance fraud detection graph, applies Node2Vec embeddings,
    clusters entities using K-Means, and visualizes potential fraudulent groups.
    """
    # Create a fraud detection graph
    G = nx.Graph()

    # Add nodes representing policyholders, claims, and adjusters
    policyholders = ["Alice", "Bob", "Charlie", "David"]
    adjusters = ["Adjuster1", "Adjuster2"]
    claims = ["ClaimA", "ClaimB", "ClaimC"]

    # Define relationships (e.g., shared addresses, previous claims, or common adjusters)
    edges = [
        ("Alice", "ClaimA"), ("Bob", "ClaimB"), ("Charlie", "ClaimC"),
        ("ClaimA", "Adjuster1"), ("ClaimB", "Adjuster1"), ("ClaimC", "Adjuster2"),
        ("Alice", "Bob"),  # Shared address
        ("Bob", "Charlie"),  # Previous suspicious claims together
        ("Charlie", "David"),  # Same employer
        ("ClaimA", "ClaimB")  # Linked claims
    ]
    
    G.add_nodes_from(policyholders, type="policyholder")
    G.add_nodes_from(adjusters, type="adjuster")
    G.add_nodes_from(claims, type="claim")
    G.add_edges_from(edges)

    # Train Node2Vec model
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Extract embeddings for nodes
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])

    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(node_embeddings)

    # Apply K-Means clustering
    num_clusters = 3  # Adjust as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced_embeddings)

    # Visualize clusters
    plt.figure(figsize=(8, 6))
    for i, node in enumerate(G.nodes()):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, c=f"C{labels[i]}", label=node)
        plt.text(x + 0.01, y + 0.01, node, fontsize=10, color=f"C{labels[i]}")

    plt.title("Insurance Fraud Clustering (K-Means + PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("08_insurance_fraud_clustering.png", dpi=300)
    print("Insurance fraud clustering plot saved as '08_insurance_fraud_clustering.png'.")

# Run function
insurance_fraud_clustering()