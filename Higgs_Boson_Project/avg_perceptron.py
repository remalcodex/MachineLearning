import numpy as np
from Perceptron import Perceptron
#import matplotlib.pyplot as plt
from sklearn import svm



def findLineCount(filename):
    linecount = 0
    f = open(filename,'r')
    for line in f:
        linecount += 1
    f.close()
    return linecount


def findN():
    linecount = 0
    maxn = 0
    f = open('phishing.train','r')
    for line in f:
        linecount+=1
        nline = line.split()
        for i in range(len(nline)):
            if i!=0:
                if int(nline[i].split(':')[0]) > maxn :
                    maxn = int(nline[i].split(':')[0])
    f.close()
    return maxn

def extractData(x,y,filename):
    count = 0
    filepointer = open(filename,'r')
    for line in filepointer:
        nline = line.split()

        for i in range(len(nline)):
            if i==0:
                y = np.append(y,int(nline[i]))
            else:
                index = int(nline[i].split(':')[0])-1
                featureval = float(nline[i].split(':')[1])
                x[count][index] = featureval
        count+=1
    filepointer.close()
    return x,y

def shuffle(linecount,x,y,p):
    arr=np.arange(linecount)
    np.random.shuffle(arr)
    x=x[arr]
    y=y[arr]
    return x,y

def train(linecount,x,y,p,epoch):
    no_updates=0
    updates_arr=np.empty([0,1])
    weight = np.empty([0,28])
    bias = np.empty([0,1])
    a=p.w
    ba=p.bias
    for i in range(epoch):
        for j in range(linecount):
            #print(x[j])
            y_pred = p.predict(x[j])
            if y_pred*y[j] < 0:
                p.update(y[j],x[j])
                no_updates+=1
            a=a+p.w
            ba=ba+p.bias
        [x,y]=shuffle(linecount,x,y,p) #changed
        weight=np.vstack((weight,a))
        bias=np.append(bias,ba)
        updates_arr=np.append(updates_arr,no_updates)
    p.w=a
    p.bias=ba
    #print (bias)
    return x,y,weight,bias,updates_arr

def findAccuracy(linecount,x,y,p):
    count=0
    rec_den = 0
    prec_den = 0
    pred = 0
    for i in range(linecount):
        y_pred = p.predict(x[i])
        if y_pred*y[i]<0:
            count+=1
        else:
            if y_pred >0:
                pred+=1
        if y_pred >0 :
            prec_den +=1
        if y[i]>0:
            rec_den+=1
    #print(y.size,p.w,x[0].size)
    return (100-count/linecount*100,pred/rec_den,pred/prec_den)

def crossValidation():
    f=[1,1,1,1,1]
    f[0] = 'training00.data'
    f[1] = 'training01.data'
    f[2] = 'training02.data'
    f[3] = 'training03.data'
    f[4] = 'training04.data'

    x=[0,0,0,0,0]
    y=[0,0,0,0,0]
    lcount=[0,0,0,0,0]

    for i in range(5):
        y[i]=np.empty([0,1])
        lcount[i]=findLineCount(f[i])
        x[i]=np.empty([lcount[i],n])
        [x[i],y[i]]=extractData(x[i],y[i],f[i])

    eta = [1,0.1,0.01]

    best = 1
    maxacc = 0

    for e in eta:
        sum=0
        for i in range(5):
            data = np.empty([0,n])
            label = np.empty([0,1])
            line_c=0
            for k in range(5):
                if k==i: continue
            else:
                data=np.vstack((data,x[k]))
                label=np.append(label,y[k])
                line_c+=lcount[k]

            cell = Perceptron(n,e)
            [data,label,w,b,updates_arr]=train(line_c,data,label,cell,10)
            acc=findAccuracy(lcount[i],x[i],y[i],cell)
            sum+=acc
        avgacc=sum/5
        if maxacc<avgacc:
            best = e
            maxacc = avgacc
        #print (maxacc)

    return best,maxacc

#--------------------------------------------------

# y = np.empty([0,1])
# n = findN()
# linecount = findLineCount('phishing.train')
# x = np.empty([linecount,n])
# [x,y]=extractData(x,y,'phishing.train')
# #print(x)
#
# y_dev = np.empty([0,1])
# linecount_dev = findLineCount('phishing.dev')
# x_dev = np.empty([linecount_dev,n])
# [x_dev,y_dev]=extractData(x_dev,y_dev,'phishing.dev')
#
# y_test = np.empty([0,1])
# linecount_test = findLineCount('phishing.test')
# x_test = np.empty([linecount_test,n])
# [x_test,y_test]=extractData(x_test,y_test,'phishing.test')

#_______________new stuff___________________
mean = [0.9914658435843994, -8.2976178820622e-06, -1.3272252572679215e-05, 0.9985363574312471, 2.6134592495411797e-05, 0.9909152318068567, -2.0275203997251415e-05, 7.71619920710906e-06, 0.9999687478206591, 0.9927294304430038, -1.0264440172703127e-05, -2.0768873493851226e-05, 1.0000080177052564, 0.9922590513707101, 1.459561349773536e-05, 3.678631990462732e-06, 1.0000114192497513, 0.9861086617144861, -5.756954065664269e-06, 1.7449033596108414e-05, 1.0000001559677123, 1.0342903040056053, 1.0248048350282475, 1.0505538681766282, 1.009741840750048, 0.972959616608593, 1.033035574431563, 0.9598119879373501]
sd=[0.5653776754096951, 1.0088264812855468, 1.006346283885119, 0.6000184644551814, 1.0063261640156402, 0.47497472589232176, 1.009302952852424, 1.0059010877868422, 1.0278075278204606, 0.49999384024846355, 1.0093304676767396, 1.0061543903728194, 1.049397999042849, 0.4876623258003873, 1.0087467092311453, 1.0063049450318349, 1.193675521568018, 0.5057776635500334, 1.0076942258109045, 1.0063655876039794, 1.4002093224446897, 0.6746353374867367, 0.38080739505009764, 0.16457624382242395, 0.39744529874617945, 0.5254062490071941, 0.3652556048435137, 0.3133377767062806]

filep = open('HIGGS/HIGGS.csv','r')
#filep = open('a.txt','r')

x = []
y =[]
count = 0
for line in filep:
    nline = line.strip().split(',')
    x.append([(float(nline[1+i])-mean[i])/sd[i] for i in range(len(nline[1:]))])
    l = int(float(nline[0]))
    if l == 1:
        y.append(l)
    else:
        y.append(-1)
    count += 1
    if count == 2500000: break
filep.close()
#filep = open('b.txt','r')
filep = open('HIGGS/HIGGS-109-test.csv','r')
x_test = []
y_test =[]
for line in filep:
    nline = line.strip().split(',')
    x_test.append([(float(nline[1+i])-mean[i])/sd[i] for i in range(len(nline[1:]))])
    l=int(float(nline[0]))
    if l == 1:
        y_test.append(l)
    else:
        y_test.append(-1)
filep.close()
#_______________new stuff___________________

x=np.array(x)
y=np.array(y)
x_test=np.array(x_test)
y_test=np.array(y_test)
#print(y[0:100000]-y_test)
# eta,bestacc=crossValidation()
# print ("Best learning rate: ",eta)
# print ('Cross Validation Accuracy for best learning rate: ',round(bestacc,2),'%')

p = Perceptron(28,0.1)
#print (p.w)
[x,y,w,b,updates_arr]=train(count,x,y,p,20)
#print(w[19])

# per = Perceptron(n,eta)
# best_epoch = 0
# maxacc = 0
#
# acc_arr=np.empty(20)
#
# for i in range(20):
# 	per.w = w[i]
# 	per.bias = b[i]
# 	acc = findAccuracy(linecount_dev,x_dev,y_dev,per)
# 	acc_arr[i]=acc
# 	if acc > maxacc:
# 		maxacc = acc
# 		best_epoch = i

#x_coordinate = np.arange(20)
# x_coordinate = [ 1 * i+1 for i in range(20) ]
# plt.plot(x_coordinate,acc_arr)
# plt.xticks(x_coordinate)
# plt.xlabel('Epoch ID')
# plt.ylabel('Development Set Accuracy')

p.w=w[19]
p.bias=b[19]
#print ('Total number of updates on training set = ',int(updates_arr[best_epoch]))
#print ('Optimal no. of epochs: ',best_epoch+1)


#print('Development Set Accuracy: ',round(maxacc,2),'%')
#print("Training Set Accuracy: ",round(findAccuracy(linecount,x,y,p),2),'%')
acc,recall,prec = findAccuracy(len(y_test),x_test,y_test,p)
print(acc,recall,prec)


# plt.title('3.2 - 4: Averaged Perceptron')
# plt.savefig('Fig_4.png')
#--------------------------------------------------
