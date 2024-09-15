import pandas as pd

file_path = "./ratings.dat"
file_dst = "./ratings_sorted.csv"

counter = 0
mapping_set = {}
# {key:value} -> {original_itemID: mapping ID}
df = pd.read_csv(file_path, delimiter='::', engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
item_num = df['movie_id'].max()


def map_itemID(movieID):
    global counter
    if movieID not in mapping_set:
        mapping_set[movieID] = counter
        counter = counter + 1
    return mapping_set[movieID]

df['processed_userID'] = df['user_id'] - 1
df['processed_itemID'] = df['movie_id'].apply(map_itemID)


df.to_csv(file_dst, index=False)