from sentence_transformers import SentenceTransformer
import torch

# 使用 GPU 加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用裝置：{device}")

# 檢查 GPU 是否可用
if torch.cuda.is_available():
    print(f"正在使用 GPU：{torch.cuda.get_device_name(0)}")
else:
    print("未使用 GPU，正在使用 CPU")