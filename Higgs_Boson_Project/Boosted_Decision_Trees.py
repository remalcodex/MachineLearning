from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
#filep = open('HIGGS-000.csv','r')

mean = [0.9914658435843994, -8.2976178820622e-06, -1.3272252572679215e-05, 0.9985363574312471, 2.6134592495411797e-05, 0.9909152318068567, -2.0275203997251415e-05, 7.71619920710906e-06, 0.9999687478206591, 0.9927294304430038, -1.0264440172703127e-05, -2.0768873493851226e-05, 1.0000080177052564, 0.9922590513707101, 1.459561349773536e-05, 3.678631990462732e-06, 1.0000114192497513, 0.9861086617144861, -5.756954065664269e-06, 1.7449033596108414e-05, 1.0000001559677123, 1.0342903040056053, 1.0248048350282475, 1.0505538681766282, 1.009741840750048, 0.972959616608593, 1.033035574431563, 0.9598119879373501]
sd=[0.5653776754096951, 1.0088264812855468, 1.006346283885119, 0.6000184644551814, 1.0063261640156402, 0.47497472589232176, 1.009302952852424, 1.0059010877868422, 1.0278075278204606, 0.49999384024846355, 1.0093304676767396, 1.0061543903728194, 1.049397999042849, 0.4876623258003873, 1.0087467092311453, 1.0063049450318349, 1.193675521568018, 0.5057776635500334, 1.0076942258109045, 1.0063655876039794, 1.4002093224446897, 0.6746353374867367, 0.38080739505009764, 0.16457624382242395, 0.39744529874617945, 0.5254062490071941, 0.3652556048435137, 0.3133377767062806]

filep = open('HIGGS/HIGGS.csv','r')
#filep = open('a.txt','r')

data = []
labels =[]
count = 0
ones=0

for line in filep:
    count += 1
    #if count<700000: continue
    nline = line.strip().split(',')
    data.append([(float(nline[1+i])-mean[i])/sd[i] for i in range(len(nline[1:]))])
    labels.append(int(float(nline[0])))
    if int(float(nline[0])) == 1:
        ones+=1
    if count == 1500000: break
    #print (count)
filep.close()
print(count)
print(ones)
#filep = open('b.txt','r')
filep = open('HIGGS/HIGGS-100-test.csv','r')
testdata = []
testlabels =[]
for line in filep:
    nline = line.strip().split(',')
    testdata.append([(float(nline[1+i])-mean[i])/sd[i] for i in range(len(nline[1:]))])
    testlabels.append(int(float(nline[0])))
filep.close()

#print(data)
print('********')
#logreg = linear_model.LogisticRegression(C=1e5)
#logreg.fit(data, labels)
clf = GradientBoostingClassifier(n_estimators=202, learning_rate=0.5, max_depth=2, random_state=0).fit(data, labels)
print ('test done')
print(clf.score(testdata,testlabels))
pred_labels = clf.predict(testdata)
#pred_labels = logreg.predict(testdata)

count = 0
recal_den = 0
prec_den = 0
pred = 0
for i in range(len(pred_labels)):
    if pred_labels[i]==testlabels[i]:
        count+=1
        if pred_labels[i]==1:
            pred+=1
    if pred_labels[i]==1:
        prec_den+=1
    if testlabels[i]==1:
        recal_den+=1                



print('Accuracy ',count/len(pred_labels))
print('RecaLL',pred/recal_den)
print('Precision ',pred/prec_den)
