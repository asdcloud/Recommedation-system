import csv
import numpy as np
import time
import random
#############################
dist = 3 #距離 整數
converge = 2 #收斂速度 數字越小收斂程度越高
dist_div = 0 # 初始化
current_timestamp = round(time.time(), 0) #知道現在的時間是多少
np.set_printoptions(precision=3)
#############################

def openmovies():
    print("opendata")
    csvfiles = open('./data/1m/movies.dat', newline='', encoding="ISO-8859-1")
    csvfiles = list(csvfiles)
    for i in range(len(csvfiles)):
        csvfiles[i] = list(csvfiles[i])
        csvfiles[i] = ''.join(csvfiles[i])
        csvfiles[i] = csvfiles[i].replace('\n','').split('::')
    return csvfiles

def tag_generate(movies_matrix):
    movie_diction = {}
    tags = []
    tags_group = []
    for i in range(len(movies_matrix)):
        movies_matrix[i].append([0, 0])
        movies_matrix[i][2] = movies_matrix[i][2].split('|')
        movie_diction[movies_matrix[i][0]] = i # md[movieid] 就是 movie在relation matrix內的位置 但問題是......我們前面會空一格 所以movieID為1的是在relatio matrix為1的位置 但其實是0
        for j in range(len(movies_matrix[i][2])):
            if movies_matrix[i][2][j] not in tags:
                tags.append(movies_matrix[i][2][j])
                empty_list = list()
                tags_group.append(empty_list)
                tags_group[len(tags_group)-1].append(i)
            else:
                tags_index = tags.index(movies_matrix[i][2][j])
                tags_group[tags_index].append(i)
    return movie_diction, tags, tags_group

def rm_generate(tags, tags_group, movies_matrix):
    print("generating relation_matrix...")
    relation_matrix = np.zeros((len(movies_matrix), len(movies_matrix)))
    for i in range(len(tags)):
        for movieA in tags_group[i]:
            for movieB in tags_group[i]:  
                relation_matrix[movieA-1][movieB-1] = relation_matrix[movieA-1][movieB-1]+1
    for i in range(len(relation_matrix)):
        relation_matrix[i][i] = 0
    max_value = np.max(relation_matrix)
    for i in range(len(relation_matrix)):#對每個元素做normalize(目前使用min-max)
        for j in range(len(relation_matrix)):
            relation_matrix[i][j] = relation_matrix[i][j]/max_value
    return relation_matrix

def KG_generate(relation_matrix):
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
        writer.writerows(result)
    return result

def openratings():
    csvfiles = open('./data/1m/ratings.dat', newline='', encoding="ISO-8859-1")
    csvfiles = list(csvfiles)
    for i in range(len(csvfiles)):
        csvfiles[i] = list(csvfiles[i])
        csvfiles[i] = ''.join(csvfiles[i])
        csvfiles[i] = csvfiles[i].replace('\n','').split('::')
    return csvfiles

def CF_train(ratings_matrix, movies_matrix, KG, movie_diction):
    for i in range(len(ratings_matrix)):
        if ratings_matrix[i][3] == "timestamp":
            continue
        rating_time = int(ratings_matrix[i][3])
        ##divation[movie_diction[ratings_matrix_rand[i][1]]] = np.append(divation[movie_diction[ratings_matrix_rand[i][1]]], float(ratings_matrix_rand[i][2]))
        print(i)
        for j in range(len(KG)):
            if (movie_diction[ratings_matrix[i][1]] == j):
                scale = round(rating_time/current_timestamp, 1)
                movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
                movies_matrix[j][3][1] += scale
            else:
                scale = round(rating_time/current_timestamp, 1) * KG[movie_diction[ratings_matrix[i][1]]][j] / len(movies_matrix) * 1
                movies_matrix[j][3][0] += (float(ratings_matrix[i][2]) * scale)
                movies_matrix[j][3][1] += scale
    return ratings_matrix, movies_matrix, movie_diction

def RMSE(movies_matrix, movie_diction, ratings_matrix):
    #count error(RMSE)
    total_error = 0
    for i in range(len(ratings_matrix)):
        movie_num = movie_diction[ratings_matrix[i][1]]
        if (movies_matrix[movie_num][3][1] == 0):
            continue
        error = float(ratings_matrix[i][2]) - (movies_matrix[movie_num][3][0]/movies_matrix[movie_num][3][1])
        total_error = total_error + error * error
    RMSE = (total_error / len(ratings_matrix))**0.5
    print("RMSE = ",RMSE)

def result_generate(movies_matrix):
    with open('./outputRating_rand.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(movies_matrix)):
            if (movies_matrix[i][3][1] == 0):
                continue
            ##popu_std = np.std(divation[i])

            writer.writerow([movies_matrix[i][0], movies_matrix[i][3][0], movies_matrix[i][3][1], movies_matrix[i][3][0]/movies_matrix[i][3][1]])

if __name__ == '__main__':
    movies_matrix = openmovies()
    movie_diction, tags, tags_group = tag_generate(movies_matrix)
    dist_div = len(movies_matrix)/converge #決定要除以多少
    relation_matrix = rm_generate(tags, tags_group, movies_matrix)
    KG = KG_generate(relation_matrix)
    ratings_matrix = openratings()

    ratings_matrix, movies_matrix, movie_diction = CF_train(ratings_matrix, movies_matrix, KG, movie_diction)
    RMSE(movies_matrix, movie_diction, ratings_matrix)
    result_generate(movies_matrix)

    ##print(tags_group)