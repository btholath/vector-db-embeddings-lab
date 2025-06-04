# vector-db-embeddings-lab
Playground for semantic search and embeddings using Pinecone vector database. Experiments, demos, and best practices for modern AI workflows.


# Indexing Search vs. Similarity Search
Indexing structures embeddings for fast access, while Similarity search uses metrics like cosine similarity to find the closest matches. These processes enable efficient and accurate data retrival in AI applications.


# Similarity Metrics
## 1. Cosine Similiarity 
Measures the angle between vectors, focusing on direction. Text analysis and high-dimensional spaces where direction is key.

## 2. Euclidean Distance
Calculates the straight-line distance between points, considering both direction and magnitude. Clustering and classification in low-dimensional spaces.

## 3. Dot Product
Computes the product of vector magnitudes and the cosine of the angle. Large-scale searches and recommendation systems with normalized vectors.


# Techniques for Efficient Retrieval
## FAISS (Facebook AI Similarity Search)
Methods used for Similarity Search: 
1. IVF (Inverted File Index) 
Partitions vector space into clusters, searching only within relevant ones.

2. HNSW (Hierarchical Navigable Small World)
Uses a multi-layered graph for efficient data search. High accuracy and fast searches, especially in high-dimensional spaces

## Annoy (Approximate Nearest Neighbors Oh Yeah)
A tree-based indexing method that uses random projections to build multiple trees, providing a fast and memory-efficient way to find approximate nearest neighbors. Optimized for scenarios where retrieval speed is crucial, event at the cost of some accuracy.


## References
https://xomnia.com/post/an-introduction-to-vector-databases-for-beginners/
