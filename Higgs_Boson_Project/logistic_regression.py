import numpy as np
from LRC import LRC


def findLineCount(filename):
	linecount = 0
	f = open(filename,'r')
	for line in f:
		linecount += 1
	f.close()
	return linecount


def findN(filename):
	maxn = 0
	f = open(filename,'r')
	for line in f:
		nline = line.split()
		for i in range(len(nline)):
			if i!=0:	
				if int(nline[i].split(':')[0]) > maxn :
					maxn = int(nline[i].split(':')[0])
	f.close()
	return maxn				

def extractData(x,y,filename):
	count = 0
	fileLRCer = open(filename,'r')
	for line in fileLRCer:
		nline = line.split()
		for i in range(len(nline)):
			if i==0:
				y = np.append(y,int(nline[i]))
				x.append({})
			else:
				index = int(nline[i].split(':')[0])-1
				x[count][index]=1
		count+=1		
	fileLRCer.close()
	return x,y

def shuffle(linecount,x,y,p):
	c = list(zip(x,y))
	np.random.shuffle(c)
	x,y=zip(*c)
	x=list(x)
	y=np.array(y)
	return x,y	

def train(linecount,x,y,p,epoch):
	for i in range(epoch):
		for j in range(linecount):
			p.update(y[j],x[j])
		[x,y]=shuffle(linecount,x,y,p) 

		
	return x,y

def findAccuracy(linecount,x,y,p):
	count=0
	for i in range(linecount):
		y_pred = p.predict(x[i])
		if y_pred*y[i]>=0:
			count+=1			
	
	return (count/linecount*100)

def crossValidation(f):
	x=[0,0,0,0]
	y=[0,0,0,0]
	lcount=[25001,25000,25000,25000]

	for i in range(4):
		y[i]=np.empty([0,1])
		#lcount[i]=findLineCount(f[i])
		x[i]=[]
		[x[i],y[i]]=extractData(x[i],y[i],f[i])

	gamma = [1,0.1,0.01,0.001,0.0001,0.00001]
	sigma = [0.1,1,10,100,1000,10000]

	best_gamma = 1
	best_sigma = 0.1
	maxacc = 0
	sumacc =0

	for g in gamma:
		for s in sigma:
			sum=0
			for i in range(4):
				#data = np.empty([0,n])
				data = []
				label = np.empty([0,1])
				line_c=0
				for k in range(4):
					if k==i: continue
				else:
					#data=np.vstack((data,x[k]))
					data += x[k]
					label=np.append(label,y[k])
					line_c+=lcount[k]
					
				cell = LRC(n,g,s)
				[data,label]=train(line_c,data,label,cell,10)
				acc=findAccuracy(lcount[i],x[i],y[i],cell)
				sum+=acc
			avgacc=sum/4	
			if maxacc<avgacc:
				best_gamma = g
				best_sigma = s
				maxacc = avgacc
			sumacc+=avgacc	
			

	return best_gamma,best_sigma,maxacc,sumacc/36

#--------------------------------------------------

#------------Input--------------------------------
train_file = 'higgs-train'
test_file = 'higgs-109-liblinear'
f=[1,1,1,1]
f[0] = 'higgs-00'
f[1] = 'higgs-01'
f[2] = 'higgs-02'
f[3] = 'higgs-03'
#-------------------------------------------------
y = np.empty([0,1])
n = 28#max(findN(train_file),findN(test_file))
linecount = 100000#findLineCount(train_file)
x=[]
[x,y]=extractData(x,y,train_file)


y_test = np.empty([0,1])
linecount_test = 100000#findLineCount(test_file)
x_test=[]
[x_test,y_test]=extractData(x_test,y_test,test_file)


gamma,sigma,bestacc,avgacc=crossValidation(f)
print ("Best gamma ",gamma)
print ("Best sigma ",sigma)
print('Average Cross Validation Accuracy',round(avgacc,2),'%')
print ('Cross Validation Accuracy for best gamma and sigma: ',round(bestacc,2),'%')

# gamma = 0.01
# sigma = 100000
p = LRC(n,gamma,sigma)

[x,y]=train(linecount,x,y,p,20)


print("Training Set Accuracy: ",round(findAccuracy(linecount,x,y,p),2),'%')

print("Test Set Accuracy: ",round(findAccuracy(linecount_test,x_test,y_test,p),2),'%')

#--------------------------------------------------


