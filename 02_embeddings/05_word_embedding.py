import numpy as np
import matplotlib.pyplot as plt

# 1. Word Embeddings (Word2Vec)
from gensim.models import Word2Vec

def word_embeddings_1():
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    model = Word2Vec(sentences, vector_size=10, window=3, min_count=1, workers=1)
    print("\nWord Embedding for 'cat':\n", model.wv['cat'])
    print("Similarity between 'cat' and 'dog':", model.wv.similarity('cat', 'dog'))

def word_embeddings_2():
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


if __name__ == "__main__":
    word_embeddings_1()
    word_embeddings_2()

