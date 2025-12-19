import pandas as pd

df = pd.read_csv('cleaned_data.csv')

# 分層隨機抽樣：每款遊戲最多抽 50 筆
sampled_df = df.groupby('game_name', group_keys=False).apply(lambda x: x.sample(n=min(len(x), 50), random_state=42))

# 儲存抽樣後的資料
sampled_df.to_csv('cleaned_data_sampled_per_game.csv', index=False)

print(f"已完成每款遊戲的分層抽樣，共抽出 {sampled_df.shape[0]} 筆資料。")
