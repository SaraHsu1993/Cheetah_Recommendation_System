import os
from sklearn.metrics import roc_auc_score
import numpy as np
#import csv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TestLabel_DIR = os.path.join(BASE_DIR, 'ArticleRec/libfm-1.40.src/article_fm_test_lable.txt')
#TestLabel_DIR = os.path.join(BASE_DIR, 'libfm-1.40.src/fm_test_lable.txt')
Result_DIR = os.path.join(BASE_DIR, 'ArticleRec/libfm-1.40.src/Result')

test_lable = open(TestLabel_DIR)
lables = []
for line in test_lable.readlines():
    line = line.strip('\n')
    lables.append(int(line))

for dirs, listdir, files in os.walk(Result_DIR):
    for file in files:
        if 'DS' in file:
            continue
        file_path = os.path.join(Result_DIR,file)
        #print file_path
        FMout = open(file_path)
        predict = []
        for line in FMout.readlines():
            line = line.strip('\n')
            #print type(line)
            predict.append(float(line))
            #predict.append(float(line))

        auc = roc_auc_score(lables, predict)
        '''convert = 0
        all = 0
        length = len(predict)
        for i in range(1,length):
            for j in range(i+1,length):
                if lables[i] <> lables[j]:
                    if (lables[i]-lables[j])*(predict[i]-predict[j])>0:
                        convert+1
                    all+1
        auc = convert/all'''
        '''csvfile = file(WRITE_DIR, 'wb')
        #csvfile = file(WRITE_DIR, 'wb')
        writer = csv.writer(csvfile)
        data = [
            auc, file_path
        ]
        writer.writerows(data)
        csvfile.close()'''

        print ('FM AUC= %f' % auc, file)