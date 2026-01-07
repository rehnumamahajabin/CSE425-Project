import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset import MusicDataset
from src.vae import VAE, train_vae, get_latent_features
from src.clustering import apply_kmeans, apply_pca_kmeans
from src.evaluation import compute_all_metrics

def main():
  print("Starting Music Clustering with Lyrics...")
  
  dataset = MusicDataset(audio_path='data/audio', 
                         lyrics_path='data/gtzan_lyrics.csv')
  
  audio_features, lyrics_texts, genre_labels, filenames = dataset.create_dataset()
  
  print(f"Audio features shape: {audio_features.shape}")
  print(f"Number of lyrics: {len(lyrics_texts)}")
  
  hybrid_features = dataset.prepare_hybrid_features(audio_features, lyrics_texts)
  print(f"Hybrid features shape: {hybrid_features.shape}")
  
  lang_labels = dataset.create_language_labels(len(genre_labels))
  
  print("Training VAE on hybrid features...")
  vae = VAE(input_dim=hybrid_features.shape[1], 
            hidden_dim=256, 
            latent_dim=32)
  vae = train_vae(vae, hybrid_features, epochs=100, lr=0.001)
  
  latent_features = get_latent_features(vae, hybrid_features)
  print(f"Latent features shape: {latent_features.shape}")
  
  print("Clustering with VAE + KMeans...")
  vae_kmeans_labels = apply_kmeans(latent_features, n_clusters=10)
  
  print("Baseline: PCA + KMeans...")
  pca_kmeans_labels = apply_pca_kmeans(hybrid_features, n_clusters=10)
  
  print("Evaluating results...")
  vae_metrics = compute_all_metrics(latent_features, vae_kmeans_labels, genre_labels)
  pca_metrics = compute_all_metrics(hybrid_features, pca_kmeans_labels, genre_labels)
  
  results_df = pd.DataFrame({
    'Method': ['VAE + KMeans', 'PCA + KMeans'],
    'Silhouette': [vae_metrics['silhouette'], pca_metrics['silhouette']],
    'Purity': [vae_metrics['purity'], pca_metrics['purity']],
    'NMI': [vae_metrics['nmi'], pca_metrics['nmi']],
    'ARI': [vae_metrics['adjusted_rand'], pca_metrics['adjusted_rand']]
  })
  
  results_df.to_csv('results/clustering_metrics.csv', index=False)
  print("Results saved to results/clustering_metrics.csv")
  
  plot_results(results_df)
  
  print("Project completed successfully!")

def plot_results(results_df):
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  
  metrics = ['Silhouette', 'Purity', 'NMI', 'ARI']
  titles = ['Silhouette Score', 'Cluster Purity', 'Normalized Mutual Info', 'Adjusted Rand Index']
  
  for idx, (metric, title) in enumerate(zip(metrics, titles)):
    row = idx // 2
    col = idx % 2
    axes[row, col].bar(results_df['Method'], results_df[metric])
    axes[row, col].set_title(title)
    axes[row, col].set_ylabel('Score')
  
  plt.tight_layout()
  plt.savefig('results/metrics_comparison.png', dpi=150)
  plt.show()

if __name__ == "__main__":
  main()