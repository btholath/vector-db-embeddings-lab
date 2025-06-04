# vector-db-embeddings-lab
Playground for semantic search and embeddings using Pinecone vector database. Experiments, demos, and best practices for modern AI workflows.



# Similarity Metrics
## Cosine Similiarity 
Measures the angle between vectors, focusing on direction. Text analysis and high-dimensional spaces where direction is key.

## Euclidean Distance
Calculates the straight-line distance between points, considering both direction and magnitude. Clustering and classification in low-dimensional spaces.

## Dot Product
Computes the product of vector magnitudes and the cosine of the angle. Large-scale searches and recommendation systems with normalized vectors.


# Techniques for Efficient Retrieval
## FAISS (Facebook AI Similarity Search)
Methods used for Similarity Search: 
1. IVF (Inverted File Index) 
Partitions vector space into clusters, searching only within relevant ones.

2. HNSW (Hierarchical Navigable Small World)
Uses a multi-layered graph for efficient data search. High accuracy and fast searches, especially in high-dimensional spaces

## References
https://xomnia.com/post/an-introduction-to-vector-databases-for-beginners/
