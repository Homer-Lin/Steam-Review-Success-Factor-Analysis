import pandas as pd
import ast

# 讀取資料檔
df_description = pd.read_csv('dataset/games_description.csv')
df_reviews = pd.read_csv('dataset/steam_game_reviews.csv')

def is_action_game(genres):
    try:
        genre_list = ast.literal_eval(genres)  # 將字串轉為清單
        return 'Action' in genre_list
    except:
        return False

df_action = df_description[df_description['genres'].apply(is_action_game)]
print(f"篩選出 Action 類型遊戲數量：{df_action.shape[0]}")
print(df_action[['name', 'genres']].head())

df_action.to_csv('dataset/action_games_description.csv', index=False)
print("Action 類型遊戲已儲存至 dataset/action_games_description.csv")

# =============== Join ================
action_games = df_action['name'].unique()

df_action_reviews = df_reviews[df_reviews['game_name'].isin(action_games)]

df_action_reviews.to_csv('dataset/action_games_reviews.csv', index=False)
print(f"過濾後的評論資料已儲存至 dataset/action_games_reviews.csv")

print(f"篩選後的評論數量：{df_action_reviews.shape[0]}")
print(df_action_reviews.head())
