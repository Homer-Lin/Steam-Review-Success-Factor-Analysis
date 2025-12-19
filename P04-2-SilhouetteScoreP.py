import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_path = "C:/Windows/Fonts/msjh.ttc"
zh_font = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = zh_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

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

sse = []
sil_scores = []
K_range = range(2, 40)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    sil_score = silhouette_score(X, kmeans.labels_)
    sil_scores.append(sil_score)

optimal_k = K_range[sil_scores.index(max(sil_scores))]
print(f'最佳群集數: {optimal_k}')

fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = 'tab:blue'
ax1.set_xlabel('K（群集數）')
ax1.set_ylabel('SSE', color=color1)
line1 = ax1.plot(K_range, sse, marker='o', color=color1, label='SSE')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:orange'
ax2.set_ylabel('Silhouette Score', color=color2)
line2 = ax2.plot(K_range, sil_scores, marker='x', color=color2, label='Silhouette Score')
ax2.tick_params(axis='y', labelcolor=color2)

# 合併圖例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.title('Elbow 與 Silhouette Score')
plt.grid(True)
plt.savefig('elbow_silhouette.png')
plt.show()
