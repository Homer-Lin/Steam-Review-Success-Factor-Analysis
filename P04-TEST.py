import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 確保下載所需 NLTK 資源
nltk.download('punkt')
nltk.download('stopwords')

# 載入清理後的資料
merged_df = pd.read_csv('cleaned_data.csv')

# 解開 sentences 欄位（字串轉 list）
all_sentences = []
for s_list in merged_df['sentences']:
    try:
        sentences = eval(s_list)
    except:
        sentences = []
    all_sentences.extend(sentences)

# 移除過短的句子
all_sentences = [s.strip() for s in all_sentences if len(s.split()) >= 3]

# TF-IDF 向量化
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=1000)
X = vectorizer.fit_transform(all_sentences)

# 自動化選擇最佳群集數
sse = []
sil_scores = []
K_range = range(2, 20)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    sil_score = silhouette_score(X, kmeans.labels_)
    sil_scores.append(sil_score)

# 找最佳群集數
optimal_k = K_range[sil_scores.index(max(sil_scores))]
print(f'最佳群集數: {optimal_k}')

# 繪製 Elbow 曲線
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o', label='SSE')
plt.plot(K_range, sil_scores, marker='x', label='Silhouette Score')
plt.xlabel('K（群集數）')
plt.ylabel('分數')
plt.title('Elbow 與 Silhouette Score')
plt.legend()
plt.grid(True)
plt.savefig('elbow_silhouette.png')
plt.show()

# KMeans 聚類分析
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X)

# 關鍵詞提取
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
keywords = {}
for i in range(optimal_k):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    keywords[f'Cluster {i}'] = top_terms

# 可視化聚類結果
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

# 儲存聚類結果與關鍵詞
clustered_data = pd.DataFrame({'sentence': all_sentences, 'cluster': labels})
clustered_data.to_csv('clustered_sentences.csv', index=False)
pd.DataFrame(keywords).to_csv('cluster_keywords.csv', index=False)
print('聚類結果已儲存為 clustered_sentences.csv 和 cluster_keywords.csv')
