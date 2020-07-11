## Main Class
## Date : 13-June-2020
## Func : Main Class

import numpy as np
import csv
from backpro import NN
import os
import glob
import sklearn.model_selection

def getUniqueList(wordList):
    uniqueList = []

    for x in (wordList):
        if x not in uniqueList :
            uniqueList.append(x)
    return uniqueList

def computeTF(wordList, uniqueList):

    wordTF = []
    lenList = len(wordList)
    for x in (uniqueList):
        wordCnt = wordList.count(x)
        TFnum = wordCnt / lenList
        wordTF.append([x, float(TFnum)])
    return wordTF
    
x = []
y = []

path = 'D:\\TA ORANG\\Angkot\\CODE\\traffic-with-BackPro\\DATA'
os.chdir(path)
files = glob.glob('*.{}'.format('csv'))

##Reading file and saving to variables

##Calculating Term Frequency for words

landmarkList = []
dayList = []

for filename in files :
    with open(filename) as csvfile:
        f_read = csv.reader(csvfile, delimiter = ',')

        for row in f_read:
            if(row[0] != 'num'):
                #Appending landmark to list
                lmWord = ''
                if any(i in row[14] for i in ','):
                    lmSplit = row[14].split(',',2)
                    lmWord = lmSplit[0]
                else:
                    lmWord = row[14]

                if(lmWord == ''):
                    landmarkList.append('-')
                else:
                    landmarkList.append(lmWord)

                #Appending Days to list
                dayList.append(row[15])

landmarkUniqueList = getUniqueList(landmarkList)
landmarkTF = computeTF(landmarkList,landmarkUniqueList)

dayUniqueList = getUniqueList(dayList)
dayTF = computeTF(dayList, dayUniqueList)

for filename in files :
    print(filename)
    with open(filename) as csvfile:
        f_read = csv.reader(csvfile, delimiter = ',')

        for row in f_read:
            if(row[0] != 'num'):
                #Labeling landmark as Number ('' as 0, 'Perumahan' as 0.1)
                lmWord = ''
                if any(i in row[14] for i in ','):
                    lmSplit = row[14].split(',',2)
                    lmWord = lmSplit[0]
                elif(row[14] == ''):
                    lmWord = '-'
                else:
                    lmWord = row[14]

                for i in (landmarkTF):
                    if(lmWord == i[0]):
                        lmWeight = i[1]
            
                #Labeling days as number
                #Old files days in row 10 | New file days are in row 15
                for i in (dayTF):
                    if(row[15] == i[0]):
                        dayWeight = i[1]

                #clean labels
                #Old files labels in row 15 | New files labels are in row 14
                if(row[13] == '-'):
                    label = 1
                elif(row[13] == 'Menurunkan Penumpang'):
                    label = 2
                elif(row[13] == 'Ngetem'):
                    label = 3
                elif(row[13] == 'Macet'):
                    label = 4

                # 3 as lat, 4 as lang, day as day, 12 as  time, landmark as landmark, label as label
                #x.append([[float(row[3]),float(row[4]),float(day),float(row[12].split(':')[0]),float(landmark)],[float(label)]])
                x.append([[float(row[3]),float(row[4]),float(dayWeight),float(row[11].split(':')[0]),float(lmWeight)],[float(label)]])
                #x = np.append(x, [float(j) for j in row]) -- Comented as only using 1 value

#split data into train and testing
trainData, testData = sklearn.model_selection.train_test_split(x, test_size=1000)

# create a network with 4 input, 4 hidden, and 4 output nodes
n = NN(5, 5, 1)
# training (x, Epoch, Learning Rate, Momentum factor)
n.train(trainData, iterations=100, N=0.1, M=0.1)
# testing
n.test(testData)
