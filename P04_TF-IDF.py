import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

merged_df = pd.read_csv('cleaned_data_sampled_per_game.csv')

all_sentences = []
for s_list in merged_df['sentences']:
    try:
        sentences = eval(s_list)
    except:
        sentences = []
    all_sentences.extend(sentences)

all_sentences = [s.strip() for s in all_sentences if len(s.split()) >= 3]

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
X = vectorizer.fit_transform(all_sentences)

optimal_k = 20  # 替換為自動化選擇的值

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
keywords = {}
for i in range(optimal_k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    keywords[f'Cluster {i}'] = top_terms

pca = PCA(n_components=2, random_state=42)
reduced_data = pca.fit_transform(X.toarray())
plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title('KMeans 聚類結果 (2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.savefig('kmeans_clusters.png')
plt.show()

clustered_data = pd.DataFrame({'sentence': all_sentences, 'cluster': labels})
clustered_data.to_csv('clustered_sentences.csv', index=False)
pd.DataFrame(keywords).to_csv('cluster_keywords.csv', index=False)
print('聚類結果已儲存為 clustered_sentences.csv 和 cluster_keywords.csv')
