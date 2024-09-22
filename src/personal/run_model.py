import torch
from model import NeuMF
from main import config
from torch import nn
import pandas as pd
from tqdm import tqdm
import warnings
import pickle
import os

# 忽略特定的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load` with `weights_only=False`")


def load_model(model_path, config, device):
    model = NeuMF(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  
    return model

# 模型路徑
model_path = './checkpoints/neumf_factor8neg4_Epoch198_HR0.6866_NDCG0.4110.model'

# 設備設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加入模型
model = load_model(model_path, config, device)

#讀取數據rating.csv
data = pd.read_csv('./res/ml-1m/ratings.csv')

# 顯示前幾行數據（裝逼用）
print("Preview of Raw Data:")
print(data.head())

# 數據預處理
user_id_mapping_file = 'user_id_mapping.pkl'
item_id_mapping_file = 'item_id_mapping.pkl'

if os.path.exists(user_id_mapping_file) and os.path.exists(item_id_mapping_file):
    with open(user_id_mapping_file, 'rb') as f:
        user_id_to_index = pickle.load(f)

    with open(item_id_mapping_file, 'rb') as f:
        item_id_to_index = pickle.load(f)
else:
    print("Mapping file not found, generating mapping file...")
    unique_user_ids = data['userId'].unique()
    user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids)}

    unique_item_ids = data['itemId'].unique()
    item_id_to_index = {item_id: index for index, item_id in enumerate(unique_item_ids)}

    # Saving the mapping as a .pkl file
    with open(user_id_mapping_file, 'wb') as f:
        pickle.dump(user_id_to_index, f)

    with open(item_id_mapping_file, 'wb') as f:
        pickle.dump(item_id_to_index, f)

    print("The mapping file has been generated and saved.")

# 檢查是否有沒被對應到的
missing_users = data[~data['userId'].isin(user_id_to_index.keys())]
missing_items = data[~data['itemId'].isin(item_id_to_index.keys())]

if not missing_users.empty:
    print("The following user IDs were not found in the mapping, prediction cannot be performed:")
    print(missing_users['userId'].unique())
    # 可以選擇從這裡刪除，不過rating已經整理過了，應該不會有這個問題
    data = data[data['userId'].isin(user_id_to_index.keys())]

if not missing_items.empty:
    print("The following item IDs were not found in the mapping, prediction cannot be performed:")
    print(missing_items['itemId'].unique())
    # 可以選擇從這裡刪除物品
    data = data[data['itemId'].isin(item_id_to_index.keys())]

# 添加索引列
data['user_index'] = data['userId'].map(user_id_to_index)
data['item_index'] = data['itemId'].map(item_id_to_index)

data = data.reset_index(drop=True)


user_indices = torch.tensor(data['user_index'].values)
item_indices = torch.tensor(data['item_index'].values)

# 批量預測
batch_size = 1024  # 怕memory爆掉

model.to(device)

predictions = []

print("Starting prediction....")

for start_idx in tqdm(range(0, len(user_indices), batch_size)):
    end_idx = min(start_idx + batch_size, len(user_indices))
    batch_user_indices = user_indices[start_idx:end_idx].to(device)
    batch_item_indices = item_indices[start_idx:end_idx].to(device)

    with torch.no_grad():
        batch_predictions = model(batch_user_indices, batch_item_indices)
        predictions.extend(batch_predictions.cpu().numpy().flatten())

# 保存預測結果
data['prediction'] = predictions

# 新的rating值（其實就是把0-1轉成1-5)
data['prediction_scaled'] = data['prediction'] * 4 + 1

# 將預測結果存到CSV
output_file = 'prediction_results.csv'
data.to_csv(output_file, index=False)

print(f"Prediction complete, the results have been saved to. {output_file}")

