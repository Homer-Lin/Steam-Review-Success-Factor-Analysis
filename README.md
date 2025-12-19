# Steam-Review-Success-Factor-Analysis (Steam 遊戲留言分析：遊戲成敗關鍵)

這個專案旨在透過分析 Steam 平台的遊戲評論，結合 **雙因子理論 (Two-Factor Theory)**，探討影響玩家對遊戲評價的「激勵因子」與「保健因子」。利用自然語言處理 (NLP) 技術與分群演算法，從大量非結構化的文字評論中挖掘出遊戲成功的關鍵要素。

## 💾 資料集來源與說明 (Dataset)

本研究使用的資料集來自 Kaggle 開源數據：

* **資料集名稱**: [Steam Games, Reviews, and Rankings](https://www.kaggle.com/datasets/mohamedtarek01234/steam-games-reviews-and-rankings)
* **來源**: Kaggle (作者: Mohamed Tarek)
* **資料規模**:
    * 包含約 **290 款** Steam 遊戲。
    * 超過 **990,000 筆** 玩家評論數據。
* **資料內容**: 包含遊戲名稱、評論內容、推薦與否、遊玩時數等欄位。本專案主要針對評論文本 (Text Review) 進行情緒分析與集群運算。

## 📊 專案分析架構

本研究將玩家評論分為兩大類進行分析：
1.  **激勵因子 (Motivator Factors)**：因為有該特徵，提升了對玩家好感（推測多出現在正向評論）。
2.  **保健因子 (Hygiene Factors)**：因為沒有該特徵，降低了對玩家好感（推測多出現在負向評論）。

## 🛠️ 技術棧 (Tech Stack)

* **Language**: Python
* **Machine Learning**: PyTorch, Scikit-learn
* **NLP Techniques**: 
    * TF-IDF (Term Frequency-Inverse Document Frequency)
    * Sentence-BERT (Semantic Search)
    * K-Means Clustering
* **Evaluation**: Elbow Method, Silhouette Score

## 📂 檔案結構說明 (File Structure)

本專案依照資料處理流程分為以下步驟：

### 1. 環境與資料準備
* `P00-torch.py`: PyTorch 環境測試。
* `P01_join_action.py`: 資料表合併與初步處理。
* `P02_datacheck.py`: 資料探索性分析 (EDA) 與檢查。
* `P03_dataclean.py`: 文字資料清理 (去除雜訊、標點符號處理)。
* `P03-01sample.py`: 資料採樣 (Sampling)。

### 2. 特徵提取 (Feature Extraction)
* `P04_TF-IDF.py`: 計算評論文本的 TF-IDF 特徵值。
* `P05-Sentence-BERT.py`: 使用預訓練模型 (SBERT) 進行語句向量化。

### 3. 模型最佳化 (Model Optimization)
* `P04-1_elbowMethod.py`: 使用手肘法 (Elbow Method) 尋找最佳分群數 $K$。
* `P04-2-SilhouetteScoreP.py`: 使用側影係數 (Silhouette Score) 評估分群效果。

### 4. 分析與洞察 (Analysis & Insights)
* `P05-1ClusterFeatures.py`: 分析各群集的關鍵字特徵。
* `P05-2Sentence.py`: 提取各群集的代表性語句。
* `P06-emotion.py`: 進行情緒分析，將群集分類為激勵或保健因子。

## 🚀 如何執行

1.  安裝必要套件：
    ```bash
    pip install torch scikit-learn transformers pandas numpy
    ```
2.  依序執行資料清理與特徵提取程式。
3.  執行 `P04-1_elbowMethod.py` 決定最佳分群數。
4.  執行 `P06-emotion.py` 產出最終情緒分析報告。
