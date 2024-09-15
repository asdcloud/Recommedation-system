import csv
import numpy as np
import time
import random
import os
import openpyxl
#參數調整
dist = 3 #距離 整數
converge = 3 #收斂速度 數字越小收斂程度越高


dist_div = 0 # 初始化
np.set_printoptions(precision=3)
csvfiles = open('./data/1m/movies.dat', newline='', encoding="ISO-8859-1")
csvfiles = list(csvfiles)
for i in range(len(csvfiles)):
    csvfiles[i] = list(csvfiles[i])
    csvfiles[i] = ''.join(csvfiles[i])
    csvfiles[i] = csvfiles[i].replace('\n','').split('::')

# 讀取 CSV 檔案內容
print("reading csvfiles...")
movies = csv.reader(csvfiles)
movies_matrix = csvfiles
tags = []
tags_group = []
print("csvfiles complete.")

movie_diction = {}


print("generating movie_matrix...")
for i in range(len(movies_matrix)):
    movies_matrix[i].append([0, 0])
    movies_matrix[i][2] = movies_matrix[i][2].split('|')
    movie_diction[movies_matrix[i][0]] = i-1 # md[movieid] 就是 movie在relation matrix內的位置 但問題是......我們前面會空一格 所以movieID為1的是在relatio matrix為1的位置 但其實是0
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
for i in range(len(relation_matrix)):
    relation_matrix[i][i] = 0
max_value = np.max(relation_matrix)
for i in range(len(relation_matrix)):#對每個元素做normalize(目前使用min-max)
    for j in range(len(relation_matrix)):
        relation_matrix[i][j] = relation_matrix[i][j]/max_value
print("relation_matrix complete")
print(len(movies_matrix))

print("generating knowledge graph...")
rm_group = list()
rm_group.append(relation_matrix)
for i in range(dist - 1):#可以重複連線幾次
    rm_group.append(np.dot(rm_group[i], relation_matrix))
    for j in range(len(relation_matrix)):#對角線歸0
        rm_group[i+1][j][j] = 0
        for k in range(len(relation_matrix)):
            rm_group[i+1][j][k] = rm_group[i+1][j][k]/dist_div

result = np.zeros((len(movies_matrix), len(movies_matrix)))
for i in range(len(relation_matrix)):
    for j in range(len(relation_matrix)):
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
csvfiles = open('./data/1m/ratings.dat', newline='', encoding="ISO-8859-1")
csvfiles = list(csvfiles)
for i in range(len(csvfiles)):
    csvfiles[i] = list(csvfiles[i])
    csvfiles[i] = ''.join(csvfiles[i])
    csvfiles[i] = csvfiles[i].replace('\n','').split('::')

ratings = csv.reader(csvfiles)
ratings_matrix = csvfiles
random_count = round(len(ratings_matrix) * 0.75, 0)
random_count = int(random_count)
ratings_matrix_rand = random.sample(ratings_matrix, random_count)

for i in range(1, len(ratings_matrix_rand)):
    if ratings_matrix_rand[i][3] == "timestamp":
        continue
    rating_time = int(ratings_matrix_rand[i][3])
    for j in range(len(relation_matrix)):
        if (movie_diction[ratings_matrix_rand[i][1]] == j):
            scale = round(rating_time/current_timestamp, 1)
            movies_matrix[j+1][3][0] += (float(ratings_matrix_rand[i][2]) * scale)
            movies_matrix[j+1][3][1] += scale
        else:
            scale = round(rating_time/current_timestamp, 1) * relation_matrix[movie_diction[ratings_matrix_rand[i][1]]][j] / len(movies_matrix) * 1
            movies_matrix[j+1][3][0] += (float(ratings_matrix_rand[i][2]) * scale)
            movies_matrix[j+1][3][1] += scale
    print(i)
print(current_timestamp)
print(len(relation_matrix))

'''
count = 0
for i in range(1, len(ratings_matrix)):
    rating_time = int(ratings_matrix[i][3])
    scale = round(rating_time/current_timestamp, 1)
    movies_matrix[movie_diction[ratings_matrix[i][1]]][3][0] += (float(ratings_matrix[i][2]) * scale)
    movies_matrix[movie_diction[ratings_matrix[i][1]]][3][1] += scale
    count += float(ratings_matrix[i][2])
    print(scale)
print(count/len(ratings_matrix))
print(current_timestamp)
'''

#count error(RMSE)
total_error = 0
for i in range(1, len(ratings_matrix)):
    movie_num = movie_diction[ratings_matrix[i][1]]+1
    if (movies_matrix[movie_num][3][1] == 0):
        continue
    error = float(ratings_matrix[i][2]) - (movies_matrix[movie_num][3][0]/movies_matrix[movie_num][3][1])
    total_error = total_error + error * error
RMSE = (total_error / len(ratings_matrix))**0.5
print("RMSE = ",RMSE)

with open('./outputRating_rand.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(1, len(movies_matrix)):
        if (movies_matrix[i][3][1] == 0):
            continue
        writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1], movies_matrix[i][3][0]/movies_matrix[i][3][1]])

'''
temparr = []
for i in range(1,len(movies_matrix)):
    temparr.append([movies_matrix[i][3][0]/movies_matrix[i][3][1], movies_matrix[i][1]])
temparr.sort()
for i in temparr:
    print(i)
'''