import csv
import numpy as np
import time
import random
import bisect

def find_position(sorted_list, target):
    index = bisect.bisect_left(sorted_list, target)
    if index < len(sorted_list) and sorted_list[index] == target:
        return index
    else:
        return -1

def find_position_in_column(matrix, column_index, target):
    # 提取該列的數據
    column_data = [row[column_index] for row in matrix]
    
    # 使用 bisect 搜尋該列
    index = bisect.bisect_left(column_data, target)
    if index < len(column_data) and column_data[index] == target:
        return index  # 回傳該數字在該列中的索引
    else:
        return -1  # 找不到則回傳 -1

def openKG():
    print("open KG")
    csvfiles = open('outputRating_rand.csv', newline='', encoding="ISO-8859-1")
    csvfiles = list(csvfiles)
    for i in range(len(csvfiles)):
        csvfiles[i] = list(csvfiles[i])
        csvfiles[i] = ''.join(csvfiles[i])
        csvfiles[i] = csvfiles[i].replace('\n','').split(',')
    csvfiles = [[float(element) for element in row] for row in csvfiles]
    return csvfiles

def openPM():
    print("open PM")
    csvfiles = open('PM_result.csv', newline='', encoding="ISO-8859-1")
    csvfiles = list(csvfiles)
    for i in range(len(csvfiles)):
        csvfiles[i] = list(csvfiles[i])
        csvfiles[i] = ''.join(csvfiles[i])
        csvfiles[i] = csvfiles[i].replace('\n','').split(',')
    csvfiles = [[float(element) for element in row] for row in csvfiles]
    return csvfiles

def openratings():
    csvfiles = open('./data/1m/ratings.dat', newline='', encoding="ISO-8859-1")
    csvfiles = list(csvfiles)
    for i in range(len(csvfiles)):
        csvfiles[i] = list(csvfiles[i])
        csvfiles[i] = ''.join(csvfiles[i])
        csvfiles[i] = csvfiles[i].replace('\n','').split('::')
    csvfiles = [[float(element) for element in row] for row in csvfiles]
    return csvfiles

def rand_split(ratings_matrix):
    random_count = round(len(ratings_matrix) * 0.25, 0)
    random_count = int(random_count)
    ratings_matrix_rand = random.sample(ratings_matrix, random_count)
    return ratings_matrix_rand

def RMSE(KG, PM, ratings_matrix):
    #count error(RMSE)
    total_error = 0
    for i in range(len(ratings_matrix)):
        userID = int(ratings_matrix[i][0])
        movieID = int(ratings_matrix[i][1])
        PM_index = find_position(PM[0], movieID)
        KG_index = find_position_in_column(KG, 0, movieID)
        val  = KG[KG_index][4]
        evaluate_rating = (val * KG[KG_index][3] +  PM[userID][PM_index]) / (1 + val)
        if(val > 1 and evaluate_rating > 3.6):
            evaluate_rating*=1
        elif(val > 1 and evaluate_rating < 3.6):
            evaluate_rating/=1
        error = float(ratings_matrix[i][2]) - evaluate_rating
        total_error = total_error + error * error
    RMSE = (total_error / len(ratings_matrix))**0.5
    print(f"RMSE = {RMSE}")

if __name__ == '__main__':
    KG = openKG()
    PM = openPM()
    ratings = openratings()
    ratings_rand = rand_split(ratings)
    RMSE(KG, PM, ratings_rand)
    

    ##print(tags_group)