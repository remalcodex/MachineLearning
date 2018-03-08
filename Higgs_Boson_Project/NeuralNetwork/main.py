import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import perceptron
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from deepNeuralNetwork import trainNeuralNetwork
from kersNetwork import trainKerasNetwork
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main():

    dataTest = pd.read_csv('HIGGS/HIGGS_1.csv', header=None)
    dataTestX = dataTest.values
    dataTestY = dataTestX[:, 0]
    #dataTestY = np.reshape(dataTestY, (dataTestY.shape[0], -1))
    dataTestX = np.delete(dataTestX, 0, 1)

    dataChunksTrain = pd.read_csv('HIGGS/HIGGS_0.csv', header=None, chunksize=2000000)
    svmPredict = svm.SVC(gamma=0.001, C=100)  # Use cross validation to find gamma.
    # eta 0.1 gave best accuracy
    model = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.01, warm_start=True)

    mean = np.array([0.9914658435843994, -8.2976178820622e-06, -1.3272252572679215e-05, 0.9985363574312471, 2.6134592495411797e-05, 0.9909152318068567, -2.0275203997251415e-05, 7.71619920710906e-06, 0.9999687478206591, 0.9927294304430038, -1.0264440172703127e-05, -2.0768873493851226e-05, 1.0000080177052564, 0.9922590513707101, 1.459561349773536e-05, 3.678631990462732e-06, 1.0000114192497513, 0.9861086617144861, -5.756954065664269e-06, 1.7449033596108414e-05, 1.0000001559677123, 1.0342903040056053, 1.0248048350282475, 1.0505538681766282, 1.009741840750048, 0.972959616608593, 1.033035574431563, 0.9598119879373501])
    stdDev = np.array([0.5653776754096951, 1.0088264812855468, 1.006346283885119, 0.6000184644551814, 1.0063261640156402, 0.47497472589232176, 1.009302952852424, 1.0059010877868422, 1.0278075278204606, 0.49999384024846355, 1.0093304676767396, 1.0061543903728194, 1.049397999042849, 0.4876623258003873, 1.0087467092311453, 1.0063049450318349, 1.193675521568018, 0.5057776635500334, 1.0076942258109045, 1.0063655876039794, 1.4002093224446897, 0.6746353374867367, 0.38080739505009764, 0.16457624382242395, 0.39744529874617945, 0.5254062490071941, 0.3652556048435137, 0.3133377767062806])
    dataTestX = dataTestX - mean
    dataTestX = dataTestX / stdDev

    counter = 0
    for chunk in dataChunksTrain:
        counter += 1
        chunkX = chunk.values
        chunkY = chunkX[:, 0]
        chunkX = np.delete(chunkX, 0, 1)
        chunkX = chunkX - mean
        chunkX = chunkX/stdDev
        trainKerasNetwork(chunkX, chunkY.copy(), dataTestX, dataTestY.copy())
        if counter == 1:
            break

if __name__ == '__main__':
    main()
