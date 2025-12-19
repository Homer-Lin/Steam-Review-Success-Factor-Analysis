import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "C:/Windows/Fonts/msjh.ttc"
zh_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = zh_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

clustered_data = pd.read_csv('clustered_sentences_bert.csv')
feature_df = pd.read_csv('cluster_features.csv')

print(f'載入聚類資料：{clustered_data.shape[0]} 筆')
print(f'載入特徵詞資料：{feature_df.shape[0]} 群集')

total_reviews = clustered_data.shape[0]
emotion_stats = clustered_data.groupby(['cluster', 'recommendation']).size().unstack(fill_value=0)
emotion_stats['total'] = emotion_stats.sum(axis=1)
emotion_stats['positive_ratio'] = emotion_stats['Recommended'] / emotion_stats['total']
emotion_stats['negative_ratio'] = emotion_stats['Not Recommended'] / emotion_stats['total']

# 加權比例計算
emotion_stats['weighted_positive_ratio'] = (emotion_stats['Recommended'] / emotion_stats['total']) * (emotion_stats['total'] / total_reviews)
emotion_stats['weighted_negative_ratio'] = (emotion_stats['Not Recommended'] / emotion_stats['total']) * (emotion_stats['total'] / total_reviews)

print('\n每個群集的正負情緒比例與加權比例：')
print(emotion_stats)

emotion_stats = emotion_stats.reset_index()
merged_stats = pd.merge(emotion_stats, feature_df, left_on='cluster', right_on='Cluster')
print('\n合併群集特徵詞和情緒比例：')
print(merged_stats.head())

output_file = 'cluster_emotion_analysis_weighted.csv'
merged_stats.to_csv(output_file, index=False)
print(f'\n分析結果已儲存至：{output_file}')

plt.figure(figsize=(10, 6))
plt.bar(merged_stats['cluster'], merged_stats['weighted_positive_ratio'], color='green', alpha=0.7, label='Weighted Positive')
plt.bar(merged_stats['cluster'], merged_stats['weighted_negative_ratio'], bottom=merged_stats['weighted_positive_ratio'], color='red', alpha=0.7, label='Weighted Negative')
plt.xlabel('Cluster')
plt.ylabel('加權比例')
plt.title('每個群集的正負情緒加權比例')
plt.legend()
plt.savefig('weighted_cluster_emotion_ratio.png')
plt.show()

print('\n圖表已儲存：weighted_cluster_emotion_ratio.png')
