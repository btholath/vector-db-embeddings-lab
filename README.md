# vector-db-embeddings-lab
Playground for semantic search and embeddings using Pinecone vector database. Experiments, demos, and best practices for modern AI workflows.



# Similarity Metrics
## Cosine Similiarity 
Description: Measures the angle between vectors, focusing on direction
Usage:  Text analysis and high-dimensional spaces where direction is key

## Euclidean Distance
Description: Calculates the straight-line distance between points, considering both direction and magnitude
Usage:  Clustering and classification in low-dimensional spaces.

## Dot Product
Description: Computes the product of vector magnitudes and the cosine of the angle.
Usage: Large-scale searches and recommendation systems with normalized vectors.


# Techniques for Efficient Retrieval
## FAISS (Facebook AI Similarity Search)
Methods used for Similarity Search: 
1. IVF (Inverted File Index) 

2. HNSW (Hierarchical Navigable Small World)
Uses a multi-layered graph for efficient data search
Merits: High accuracy and fast searches, especially in high-dimensional spaces

## References
https://xomnia.com/post/an-introduction-to-vector-databases-for-beginners/
