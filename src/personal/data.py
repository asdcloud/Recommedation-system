import pandas as pd
import random
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class UserItemRatingsData(Dataset):
    def __init__(self, userTensor, itemTensor, ratingsTensor):
        self.userTensor = userTensor
        self.itemTensor = itemTensor
        self.ratingsTensor = ratingsTensor
    def __getitem__(self, index):
        return (self.userTensor[index], self.itemTensor[index], self.ratingsTensor[index])
    def __len__(self):
        assert len(self.userTensor) == len(self.itemTensor) == len(self.ratingsTensor), "The lengths of tensor are not consistent !"
        return len(self.userTensor)


class PreprocessData:
    def __init__(self, ratings, datatype):
        self.ratings = ratings
        self.user_set = set(self.ratings['userId'].unique()) # list all user
        self.item_set = set(self.ratings['itemId'].unique()) # list all item
        self.negative_case = self._negative_set(ratings)
        
        if datatype == "implicit":
            self.preprocess_ratings = self._binarize(ratings)
        else:
            self.preprocess_ratings = self._normalize(ratings)
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
            
    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        implicit_data = ratings
        implicit_data.loc[implicit_data['rating'] > 0, 'rating'] = 1.0

        return implicit_data

    def _normalize(self, ratings):
        """explicit feedback, mapping 0 to 5 into range [0, 1]"""
        explicit_data = ratings
        max_rating = explicit_data['rating'].max()
    
        return explicit_data

    # if user and item not interact, put them into set
    # all - interact = not interact, return negative set
    def _negative_set(self, ratings):
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'}
        )
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_set - x)
        # 在模型訓練過程中創建 對比數據，以幫助模型學習區分哪些物品是用戶可能感興趣的（正樣本），哪些是不感興趣的（負樣本）
        # 這個數字在實驗後期可能需要做一些更動來對比哪個數字的效果會比較好
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(list(x), 99))
        # print(interact_status)
        # make sure all negative samples are in negative items
        # print(interact_status.apply(
        #     lambda row : all(elem in row['negative_items'] for elem in row['negative_samples']), axis=1
        #     )
        # )
        return interact_status[['userId', 'negative_items', 'negative_samples']]
    

    def instance_a_train_loader(self, num_negatives:int, batch_size:int):
        users, items, ratings = [], [], []  # list to tensor
        train_ratings = pd.merge(self.train_ratings, self.negative_case[['userId', 'negative_items']], on='userId')
        print('finishing on merging')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(list(x), num_negatives))
        # print(train_ratings)

        # 這邊加入 positive 以及 negative case
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):          
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        print('finished')
        dataset = UserItemRatingsData(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]

# """
# test section below:

# # """
# data_path = "./res/ml-1m/ratings.csv"
# data = pd.read_csv(data_path)
# # print(data.head())
# test = PreprocessData(data, "implicit").instance_a_train_loader(3, 99)

# # test._negative_interaction(data)