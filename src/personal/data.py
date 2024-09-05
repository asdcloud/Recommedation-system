import pandas as pd
import random
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader


class PreprocessData:
    def __init__(self, ratings, datatype):
        self.ratings = ratings
        self.user_set = set(self.ratings['userId'].unique()) # list all user
        self.item_set = set(self.ratings['itemId'].unique()) # list all item

        if datatype == "implicit":
            self.preprocess_ratings = self._binarize(ratings)
        else:
            self.preprocess_ratings = self._normalize(ratings)


    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        implicit_data = ratings
        implicit_data.loc[implicit_data['rating'] > 0, 'rating'] = 1.0
        print(implicit_data)
        return implicit_data

    def _normalize(self, ratings):
        """explicit feedback, mapping 0 to 5 into range [0, 1]"""
        explicit_data = ratings
        max_rating = explicit_data['rating'].max()
        

        return explicit_data

