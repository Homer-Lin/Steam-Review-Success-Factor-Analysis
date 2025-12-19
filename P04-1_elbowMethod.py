import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

# Elbow Method
sse = []
K_range = range(2, 40)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# 繪製 Elbow 曲線
plt.figure(figsize=(8, 5))
plt.plot(K_range, sse, marker='o')
plt.xticks(K_range)
plt.xlabel('K（群集數）')
plt.ylabel('SSE（誤差平方和）')
plt.title('Elbow Method 找最佳主題數')
plt.grid(True)
plt.show()


