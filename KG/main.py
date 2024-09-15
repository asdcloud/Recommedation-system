import csv
import numpy as np
import time
import random
import os
import openpyxl
import pandas as pd
#參數調整
dist = 3 #距離 整數
converge = 3 #收斂速度 數字越小收斂程度越高


dist_div = 0 # 初始化
np.set_printoptions(precision=3)
file = open('./data/100k/u.item', encoding="ISO-8859-1")
movies = list(file)
for i in range(len(movies)):
    movies[i].split('|')
for i in range(len(movies)):
    a = list(movies[i])
    a = ''.join(a)
    a = a.replace('\n','').split('|')
    print(a)