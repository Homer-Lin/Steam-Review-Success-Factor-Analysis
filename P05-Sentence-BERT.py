import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import font_manager

# font_path = "C:/Windows/Fonts/msjh.ttc"
# zh_font = font_manager.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = zh_font.get_name()
# plt.rcParams['axes.unicode_minus'] = False

input_file = 'cleaned_data_sampled_per_game.csv'
df = pd.read_csv(input_file)
print(f"載入評論資料：{df.shape[0]} 筆")

# 使用 GPU 加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用裝置：{device}")

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("正在計算句子向量，請稍候...")

embeddings = model.encode(df['clean_review'].tolist(), device=device, batch_size=64, show_progress_bar=True)
print(f"嵌入特徵完成：{embeddings.shape}")

np.save('sentence_embedding.npy', embeddings)


# sse = []
# sil_scores = []
# K_range = range(6, 40)  # 調整 K 值範圍

# for k in K_range:
#     print(f"正在聚類，K={k}...")
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     labels = kmeans.fit_predict(embeddings)

# #     # 計算 SSE 和 Silhouette Score
#     sse.append(kmeans.inertia_)
#     try:
#         sil_score = silhouette_score(embeddings, labels)
#         sil_scores.append(sil_score)
#         print(f"K={k}, Silhouette Score={sil_score:.4f}")
#     except ValueError as e:
#         print(f"K={k}, 無法計算 Silhouette Score：{e}")
#         sil_scores.append(0)  # 如果無法計算，記為 0

# optimal_k = K_range[sil_scores.index(max(sil_scores))]
optimal_k = 10
# print(f"\n最佳群集數：K={optimal_k}, Silhouette Score={max(sil_scores):.4f}")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
final_labels = final_kmeans.fit_predict(embeddings)

# 加入群集標籤至資料框
df['cluster'] = final_labels

# 儲存聚類結果
output_file = 'clustered_sentences_bert.csv'
df.to_csv(output_file, index=False)
print(f"聚類結果已儲存至：{output_file}")

# fig, ax1 = plt.subplots(figsize=(8, 5))

# # SSE 曲線
# color1 = 'tab:blue'
# ax1.set_xlabel('K（群集數）')
# ax1.set_ylabel('SSE', color=color1)
# ax1.plot(K_range, sse, marker='o', color=color1, label='SSE')
# ax1.tick_params(axis='y', labelcolor=color1)

# # # Silhouette Score 曲線
# ax2 = ax1.twinx()
# color2 = 'tab:orange'
# ax2.set_ylabel('Silhouette Score', color=color2)
# ax2.plot(K_range, sil_scores, marker='x', color=color2, label='Silhouette Score')
# ax2.tick_params(axis='y', labelcolor=color2)

# fig.tight_layout()
# plt.title('KMeans 聚類：SSE 與 Silhouette Score ')
# plt.show()

# plt.savefig('kmeans_sentence_bert.png')
# print("圖表已儲存為：kmeans_sentence_bert.png")
