#import pandas as pd
#from sklearn import svm
import math
#filep = open('HIGGS-000.csv','r')
# filep = open('HIGGS.csv','r')

# count = 0
# #= [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
# suml= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# for line in filep:
#     count += 1
    
#     nline = line.strip().split(',')
#     #data.append([float(i) for i in nline[1:]])
#     for i in range(len(suml)):
#         suml[i]+=float(nline[1+i])
           
#     # labels.append(int(float(nline[0])))
#     # if int(float(nline[0])) == 1:
#     #     ones+=1
#     # if count == 200000: break
#     #print (count)
# filep.close()
# for i in range(len(suml)):
#     suml[i]=suml[i]/count

filep = open('HIGGS-109.csv','r')
testdata = []
testlabels =[]
one=0

for line in filep:
    nline = line.strip().split(',')
    #testdata.append([(float(nline[1+i])-mean[i])/sd[i] for i in range(len(nline[1:]))])
    #testlabels.append(int(float(nline[0])))
    if int((float(nline[0])))==1:
        one+=1
filep.close()




print(one)


