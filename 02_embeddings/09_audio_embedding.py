import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def audio_embeddings(audio_files):
    """
    Extracts audio embeddings using MFCCs from multiple files and visualizes them using t-SNE.
    Saves the scatter plot as 'audio_embeddings_tsne.png'.
    """
    embeddings = []

    for audio_path in audio_files:
        # Load the audio file
        y, sr = librosa.load(audio_path)

        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_embedding = np.mean(mfccs.T, axis=0)
        embeddings.append(mfcc_embedding)

        print(f"Audio Embedding for {audio_path}: {mfcc_embedding}")

    # Convert embeddings list to NumPy array
    embeddings = np.array(embeddings)

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)  # Perplexity < number of samples
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Visualize embeddings in 2D
    plt.figure(figsize=(8, 6))
    for i, audio_path in enumerate(audio_files):
        x, y = reduced_embeddings[i]
        plt.scatter(x, y, label=audio_path)
        plt.text(x + 0.01, y + 0.01, audio_path, fontsize=10)

    plt.title("Audio Embeddings Visualization (t-SNE)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig("audio_embeddings_tsne.png", dpi=300)
    print("Audio embeddings plot saved as 'audio_embeddings_tsne.png'.")

# Example usage with multiple audio files
audio_files = ["1.mp3", "1.wav"]  # Add more audio files as needed
audio_embeddings(audio_files)