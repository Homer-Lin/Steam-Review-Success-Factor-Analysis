import pandas as pd


action_games_joined = "dataset\\action_games_description.csv"
action_games_reviews = "dataset\\action_games_reviews.csv"

action_games_joined = pd.read_csv(action_games_joined)
action_games_reviews = pd.read_csv(action_games_reviews)

print("=== 遊戲資訊===")
print(action_games_joined.head())
print(action_games_joined.info())

print("\n=== 玩家評論 (steam_game_reviews) ===")
print(action_games_reviews.head())
print(action_games_reviews.info())

initial_count = action_games_reviews.shape[0]
action_games_reviews.drop_duplicates(inplace=True)
print(f"\n評論資料：從 {initial_count} 筆移除重複後剩 {action_games_reviews.shape[0]} 筆。")

print("\nsteam_reviews 缺失值統計：")
print(action_games_reviews[['review', 'recommendation']].isnull().sum())

steam_reviews_clean = action_games_reviews.dropna(subset=['review', 'recommendation'])
print(f"\n清理後評論資料筆數：{steam_reviews_clean.shape[0]}")

merged_df = pd.merge(steam_reviews_clean, action_games_joined, left_on='game_name', right_on='name', how='inner')
print(f"\n合併後資料筆數：{merged_df.shape[0]}")

output_filename = "processed_data.csv"
merged_df.to_csv(output_filename, index=False)
print(f"\n處理後的資料已儲存至 {output_filename}")