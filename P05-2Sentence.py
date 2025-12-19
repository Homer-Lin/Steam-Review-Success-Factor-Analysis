import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances

df = pd.read_csv('cleaned_data_sampled_per_game.csv')
embeddings = np.load('sentence_embedding.npy')

optimal_k = 10 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(embeddings)
df['cluster'] = labels

representative_sentences = []

for cluster_id in range(optimal_k):
    cluster_indices = np.where(labels == cluster_id)[0]  # 找出此群的句子 index
    cluster_embeddings = embeddings[cluster_indices]     # 該群的向量
    center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

    distances = cosine_distances(cluster_embeddings, center).flatten()
    closest_idx = cluster_indices[np.argmin(distances)]  # 最接近中心者

    representative_sentences.append(df.iloc[closest_idx]['clean_review'])

summary_df = pd.DataFrame({
    'Cluster': range(optimal_k),
    'Representative_Sentence': representative_sentences
})

summary_df.to_csv('cluster_representative_sentences.csv', index=False)
print("已儲存代表句總覽至 cluster_representative_sentences.csv")
