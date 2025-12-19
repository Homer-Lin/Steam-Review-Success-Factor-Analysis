import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

merged_df = pd.read_csv('processed_data.csv')
print("已載入 processed_data.csv，資料形狀：", merged_df.shape)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # 轉換為小寫
    text = re.sub(r'<[^>]+>', '', text)  # 移除 HTML 標籤
    text = re.sub(r'(?<=\w)-(?=\w)', ' - ', text)  # 確保破折號兩側有空格
    text = re.sub(r'\betc\.', 'etc.', text)  # 確保 etc. 保持完整
    text = re.sub(r'\s+', ' ', text).strip()  # 移除多餘的空格
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # 移除非 ASCII 字符
    text = re.sub(r'(?<=[.!?])(?=[a-zA-Z])', ' ', text)  # 確保標點後有空格
    words = word_tokenize(text)
    
    # 去除停用詞
    words = [word for word in words if word not in stop_words]

    # 詞形還原（Lemmatization）
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words) 

merged_df['clean_review'] = merged_df['review'].apply(clean_text)

# 刪除重複評論
merged_df = merged_df.drop_duplicates(subset=['clean_review'], keep='first')

def remove_spam(text):
    text = re.sub(r'(.)\1{4,}', r'\1', text)  # 移除超過 4 次的連續字元
    return text

merged_df['clean_review'] = merged_df['clean_review'].apply(remove_spam)

def is_meaningful_review(text):
    if re.match(r'^[^\w]+$', text):  # 只包含標點符號
        return False
    if len(text.split()) < 5:  # 低於 5 個單字
        return False
    return True

merged_df = merged_df[merged_df['clean_review'].apply(is_meaningful_review)]

# 句子切分
def split_sentences(text):
    sentences = sent_tokenize(text)
    final_sentences = []
    for sent in sentences:
        sub_sentences = re.split(r'[-]', sent) 
        final_sentences.extend([s.strip() for s in sub_sentences if s.strip()])
    return final_sentences

merged_df['sentences'] = merged_df['clean_review'].apply(split_sentences)

print("處理後的評論內容：")
print(merged_df[['review', 'clean_review', 'sentences']].head())

merged_df.to_csv('cleaned_data.csv', index=False)
print("清理後的完整資料已儲存為 cleaned_data.csv，資料形狀：", merged_df.shape)

merged_df.head(100).to_csv('cleaned_data_top100.csv', index=False)
print("前 100 筆資料已儲存為 cleaned_data_top100.csv，資料形狀：", merged_df.head(100).shape)

merged_df.head(10).to_csv('cleaned_data_top10.csv', index=False)
print("前 10 筆資料已儲存為 cleaned_data_top10.csv，資料形狀：", merged_df.head(10).shape)
