import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

clustered_data = pd.read_csv('clustered_sentences_bert.csv')
print(f"載入聚類資料：{clustered_data.shape[0]} 筆")

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

cluster_features = {}

for cluster_num in sorted(clustered_data['cluster'].unique()):
    cluster_texts = clustered_data[clustered_data['cluster'] == cluster_num]['clean_review'].tolist()
    print(f"處理群集 {cluster_num}，評論數量：{len(cluster_texts)}")

    # 合併所有評論為單一文本
    combined_text = ' '.join(cluster_texts)

    # 計算 TF-IDF
    tfidf_matrix = vectorizer.fit_transform([combined_text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # 建立特徵詞表
    feature_dict = dict(zip(feature_names, tfidf_scores))
    sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)

    # 取得前 10 大特徵詞
    top_features = [word for word, score in sorted_features[:10]]
    cluster_features[cluster_num] = top_features
    print(f"群集 {cluster_num} 特徵詞：{', '.join(top_features)}")

feature_df = pd.DataFrame.from_dict(cluster_features, orient='index', columns=[f'Feature_{i+1}' for i in range(10)])
feature_df.to_csv('cluster_features.csv', index_label='Cluster')
print("\n群集特徵詞已儲存至 cluster_features.csv")
