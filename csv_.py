import cv2
import os
import numpy as np
import pandas as pd 
import csv
import re
import pickle
I = []
for filename in os.listdir(r'E:\Fastorch-main\train_test_256\train'):
    classname=re.findall('\w+_[0-9]',filename)[0][:-2]
    # for i,classname in enumerate(classname)
    #     namelabel[classname] = i;
    I.append(classname)
s1 = set(I)
# print(I)
# for i,j in enumerate(s1,1):
#     print(i,j)

a = list(enumerate(s1,start=0))
dic = dict(a)
print(dic)
# f=open("train.csv","w")
# for line in s1:
#     f.write(line+'\n')
# f.close()
# dic={}
# dic[j] = i
# print(dic)
with open('train.pkl','wb') as f:
    pickle.dump(dic,f)