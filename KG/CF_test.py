import csv
import numpy as np
import time
import os
import openpyxl
#參數調整
dist = 3 #距離 整數
converge = 3 #收斂速度 數字越小收斂程度越高


dist_div = 0 # 初始化
np.set_printoptions(precision=3)
csvfiles = open('./data/moviesTest.csv', newline='', encoding="utf-8")

# 讀取 CSV 檔案內容
print("reading csvfiles...")
movies = csv.reader(csvfiles)
movies_matrix = list(movies)
tags = []
tags_group = []
print("csvfiles complete.")

movie_diction = {}


print("generating movie_matrix...")
for i in range(len(movies_matrix)):
    movies_matrix[i].append([0, 0])
    movies_matrix[i][2] = movies_matrix[i][2].split('|')
    movie_diction[movies_matrix[i][0]] = i
    for j in range(len(movies_matrix[i][2])):
        if movies_matrix[i][2][j] not in tags:
            tags.append(movies_matrix[i][2][j])
            empty_list = list()
            tags_group.append(empty_list)
            tags_group[len(tags_group)-1].append(i)
        else:
            tags_index = tags.index(movies_matrix[i][2][j])
            tags_group[tags_index].append(i)
print("movie_matrix complete")

dist_div = len(movies_matrix)/converge #決定要除以多少
relation_matrix = np.zeros((len(movies_matrix)-1, len(movies_matrix)-1))


print("generating relation_matrix...")
for i in range(1, len(tags)):
    for movieA in tags_group[i]:
        for movieB in tags_group[i]:  
            relation_matrix[movieA-1][movieB-1] = relation_matrix[movieA-1][movieB-1]+1
for i in range(len(movies_matrix) - 1):
    relation_matrix[i][i] = 0
max_value = np.max(relation_matrix)
for i in range(len(movies_matrix) - 1):#對每個元素做normalize(目前使用min-max)
    for j in range(len(movies_matrix) - 1):
        relation_matrix[i][j] = relation_matrix[i][j]/max_value
print("relation_matrix complete")


print("generating knowledge graph...")
rm_group = list()
rm_group.append(relation_matrix)
for i in range(dist - 1):#可以重複連線幾次
    rm_group.append(np.dot(rm_group[i], relation_matrix))
    for j in range(len(movies_matrix) - 1):#對角線歸0
        rm_group[i+1][j][j] = 0
        for k in range(len(movies_matrix) - 1):
            rm_group[i+1][j][k] = rm_group[i+1][j][k]/dist_div

result = np.zeros((len(movies_matrix)-1, len(movies_matrix)-1))
for i in range(len(movies_matrix) - 1):
    for j in range(len(movies_matrix) - 1):
        rm_max = []
        for k in range(dist):
            rm_max.append(rm_group[k][i][j])
        result[i][j] = max(rm_max) #取最大值
print("knowledge graph complete\n")
print("the result is :")
print(result)

with open('./output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

  # 寫入二維表格
    writer.writerows(result)

#接下來是CF部分 
current_timestamp = round(time.time(), 0) #知道現在的時間是多少
csvfiles = open('./data/ratings.csv', newline='', encoding="utf-8")

ratings = csv.reader(csvfiles)
ratings_matrix = list(ratings)
#'''
for i in range(1, len(ratings_matrix)):
    rating_time = int(ratings_matrix[i][3])
    for j in range(len(relation_matrix)):
        if (movie_diction[ratings_matrix[i][1]] == j):
            scale = round(rating_time/current_timestamp, 1)
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
        else:
            scale = round(rating_time/current_timestamp, 1) * relation_matrix[movie_diction[ratings_matrix[i][1]]-1][j] / len(movies_matrix) * 6
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
    print(i)
print(current_timestamp)
print(len(relation_matrix))
#'''

with open('./outputRating01.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1, len(movies_matrix)):
        writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1]])

current_timestamp = round(time.time(), 0) #知道現在的時間是多少
csvfiles = open('./data/ratings.csv', newline='', encoding="utf-8")

ratings = csv.reader(csvfiles)
ratings_matrix = list(ratings)
#'''
for i in range(1, len(ratings_matrix)):
    rating_time = int(ratings_matrix[i][3])
    for j in range(len(relation_matrix)):
        if (movie_diction[ratings_matrix[i][1]] == j):
            scale = round(rating_time/current_timestamp, 1)
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
        else:
            scale = round(rating_time/current_timestamp, 1) * relation_matrix[movie_diction[ratings_matrix[i][1]]-1][j] / len(movies_matrix) * 7
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
    print(i)
print(current_timestamp)
print(len(relation_matrix))
#'''

with open('./outputRating02.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1, len(movies_matrix)):
        writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1]])

        current_timestamp = round(time.time(), 0) #知道現在的時間是多少
csvfiles = open('./data/ratings.csv', newline='', encoding="utf-8")

ratings = csv.reader(csvfiles)
ratings_matrix = list(ratings)
#'''
for i in range(1, len(ratings_matrix)):
    rating_time = int(ratings_matrix[i][3])
    for j in range(len(relation_matrix)):
        if (movie_diction[ratings_matrix[i][1]] == j):
            scale = round(rating_time/current_timestamp, 1)
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
        else:
            scale = round(rating_time/current_timestamp, 1) * relation_matrix[movie_diction[ratings_matrix[i][1]]-1][j] / len(movies_matrix) * 8
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
    print(i)
print(current_timestamp)
print(len(relation_matrix))
#'''

with open('./outputRating03.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1, len(movies_matrix)):
        writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1]])

        current_timestamp = round(time.time(), 0) #知道現在的時間是多少
csvfiles = open('./data/ratings.csv', newline='', encoding="utf-8")

ratings = csv.reader(csvfiles)
ratings_matrix = list(ratings)
#'''
for i in range(1, len(ratings_matrix)):
    rating_time = int(ratings_matrix[i][3])
    for j in range(len(relation_matrix)):
        if (movie_diction[ratings_matrix[i][1]] == j):
            scale = round(rating_time/current_timestamp, 1)
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
        else:
            scale = round(rating_time/current_timestamp, 1) * relation_matrix[movie_diction[ratings_matrix[i][1]]-1][j] / len(movies_matrix) * 9
            movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
            movies_matrix[j][3][1] += scale
    print(i)
print(current_timestamp)
print(len(relation_matrix))
#'''

with open('./outputRating04.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1, len(movies_matrix)):
        writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1]])
