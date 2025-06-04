"""
simple 2D embeddings visualization—this time, for popular Italian (and Italy-loved) cars in 2025. 
We'll assign each car a “2D embedding” just for illustrative purposes (randomized but nicely spaced), and label each on the plot.
"""

# /02_embeddings/italian_cars_embedding.py
import numpy as np
import matplotlib.pyplot as plt

# Let's create simple 2D embeddings for a few notable cars in Italy
car_embeddings = {
    "Fiat 500e": [0.8, 0.9],
    "Alfa Romeo Junior": [0.7, 0.7],
    "Dacia Sandero": [0.2, 0.5],
    "Tesla Model Y": [0.4, 0.95],
    "Ferrari SF90 XX Stradale": [0.95, 0.3],
    "Lamborghini Revuelto": [0.99, 0.1]
}

# Plotting the embeddings
plt.figure(figsize=(10, 8))
for car, embedding in car_embeddings.items():
    plt.scatter(embedding[0], embedding[1], label=car)
    plt.annotate(car, (embedding[0]+0.01, embedding[1]+0.01))

plt.title("Popular Cars in Italy (2025) - 2D Embedding Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Print out the embeddings
for car, embedding in car_embeddings.items():
    print(f"{car}: {embedding}")
