import numpy as np
import random

def getNumOfFeatures(inFile):
    noOfFeatures = 0;
    noOfExamples = 0
    for line in inFile:
        values = line.split()
        features = int(values[-1].split(':')[0])
        if noOfFeatures < features:
            noOfFeatures = features
        noOfExamples += 1
    inFile.seek(0)
    return noOfFeatures, noOfExamples

def readData(inFilePath, maxNoOfFeatures):

    #my_path = os.path.abspath(os.path.dirname(__file__))
    #path = os.path.join(my_path, inTrainingFileName)
    trainingFile = open(inFilePath, "r", encoding="utf8")

    noOfFeatures, noOfExamples = getNumOfFeatures(trainingFile)

    X = np.zeros((noOfExamples, maxNoOfFeatures), 'double')
    Y = np.zeros((noOfExamples, 1), int)

    counter = 0
    for line in trainingFile:
        values = line.split()
        #Adding the label.
        Y[counter] = int(values[0])

        #Adding the features.
        x = np.array([0]*maxNoOfFeatures, 'float')
        for value in values[1:]:
            x[int(value.split(':')[0])-1] = float(value.split(':')[1])
        X[counter] = x
        counter += 1

    return X, Y

#Code for training the classifier.
def trainClassifier(X, Y, weight, bias, eta, averagedWeight, averagedBias):

    totalUpdates = 0;
    for row, y in zip(X, Y):
        prediction = (weight.dot(row) + bias);
        if y*prediction < 0:
            weight = weight + eta*y*row
            bias = bias + eta*y
        totalUpdates += 1
        averagedWeight += weight
        averagedBias += bias
    return weight, bias, totalUpdates, averagedWeight, averagedBias

def predict(X, Y, weight, bias):

    Ypredict = X.dot(weight.transpose())+ bias
    #Ypredict = np.sign(Ypredict)
    Ypredict[Ypredict >= 0] = 1
    Ypredict[Ypredict < 0] = -1
    labels, counts = np.unique((Ypredict == Y), return_counts=True)
    total = counts[0] + counts[1]

    accuracy = -1;
    for label,count in zip(labels, counts):
        if label == True:
            accuracy = count/total;

    #print('Accuracy' + str(accuracy))
    return float(accuracy*100)

def chooseBestHyperParameter():
    # Checking max number of features.
    maxNoOfFeatures = 0;  # This is used later on since many files have features less than the max.
    for i in range(5):
        fileName = 'Dataset/CVSplits/training0' + str(i) + '.data'
        noOfFeatures, noOfExamples = getNumOfFeatures(open(fileName, "r", encoding="utf8"))
        if maxNoOfFeatures < noOfFeatures:
            maxNoOfFeatures = noOfFeatures

    # Getting the total dataset.
    listX = []
    listY = []
    for i in range(5):
        fileName = 'Dataset/CVSplits/training0' + str(i) + '.data'
        X, Y = readData(fileName, maxNoOfFeatures)
        listX.append(X)
        listY.append(Y)

    counter = 0
    etaAccuracy = np.array([0.0, 0.0, 0.0], 'float')
    etas = np.array([1, 0.1, 0.01], 'float')
    for eta in etas:
        avgAccuracy = 0.0
        for keepOut in range(5):
            Xtrain = np.empty((0, 1), 'float')
            Ytrain = np.empty((0, 1), 'int')
            Xcv = []
            Ycv = []

            for i in range(5):
                if i == keepOut:
                    Xcv = listX[i]
                    Ycv = listY[i]
                else:
                    if Xtrain.size == 0:
                        Xtrain = listX[i]
                        Ytrain = listY[i]
                    else:
                        Xtrain = np.vstack((Xtrain, listX[i]))
                        Ytrain = np.vstack((Ytrain, listY[i]))

            #Training the classifier.
            fWeight = np.random.rand(1, noOfFeatures)
            fWeight = (fWeight * 0.02) - 0.01
            fBias = random.uniform(-0.01, 0.01)
            averageWeight = np.zeros((1, noOfFeatures))
            averageBias = 0
            for epoch in range(10):
                # Randomizing the training set.
                Xtrain = np.array(Xtrain)
                Ytrain = np.array(Ytrain)
                randomIndices = np.arange(Xtrain.shape[0])
                np.random.shuffle(randomIndices)
                Xtrain = Xtrain[randomIndices]
                Ytrain = Ytrain[randomIndices]

                # Training the classifier.
                fWeight, fBias, updates, averageWeight, averageBias = trainClassifier(Xtrain, Ytrain, fWeight, fBias, eta, averageWeight, averageBias)

            accuracy = predict(Xcv, Ycv, averageWeight, averageBias)
            avgAccuracy += accuracy
        avgAccuracy = avgAccuracy / (5)
        etaAccuracy[counter] = avgAccuracy
        counter += 1
        # print("Eta accuracy: " + str(avgAccuracy))

    # Getting max accuracy:
    accIndex = np.argmax(etaAccuracy)
    print('The best hyper-parameters are: ' + str(etas[accIndex]))
    print('The cross validation accuracy for the best hyperparameter is: ' + str(round(etaAccuracy[accIndex],2)) + '%')
    return etas[accIndex], etaAccuracy[accIndex]

def main():

    #Choosing the hyperparameter
    eta, etaAccuracy = chooseBestHyperParameter()

    maxNoOfFeatures = 0
    fileName = 'Dataset/phishing.train'
    maxNoOfFeatures, noOfExamples = getNumOfFeatures(open(fileName, "r", encoding="utf8"));

    #Reading training set.
    fileName = 'Dataset/phishing.train'
    Xtrain, Ytrain = readData(fileName, maxNoOfFeatures)

    #Reading dev set.
    fileName = 'Dataset/phishing.dev'
    Xdev, Ydev = readData(fileName, maxNoOfFeatures)

    epochs = []
    accuracies = []
    weights = []
    biasArray = []
    fWeight = np.random.rand(1, maxNoOfFeatures)
    fWeight = (fWeight * 0.02) - 0.01
    fBias = random.uniform(-0.01, 0.01)
    averageWeight = np.zeros((1, maxNoOfFeatures))
    averageBias = 0
    updatesArray = []
    for epoch in range(20):
        # Randomizing the training set.
        randomIndices = np.arange(Xtrain.shape[0])
        np.random.shuffle(randomIndices)
        Xtrain = Xtrain[randomIndices]
        Ytrain = Ytrain[randomIndices]

        # Training the classifier on the train set.
        fWeight, fBias, updates, averageWeight, averageBias = trainClassifier(Xtrain, Ytrain, fWeight, fBias, eta, averageWeight, averageBias)
        weights.append(averageWeight)
        biasArray.append(averageBias)
        updatesArray.append(updates)

        #Predicting the classifier on the test set.
        accuracy = predict(Xdev, Ydev, averageWeight, averageBias)
        epochs.append(epoch)
        accuracies.append(accuracy)

    #Selecting the best weight
    accuracies = np.array(accuracies)
    index = np.argmax(accuracies)
    bestWeight = weights[index]
    bestBias = biasArray[index]
    bestAccuracy = accuracies[index]
    bestUpdate = updatesArray[index]
    print('Total number of updates on training set: ' + str(bestUpdate))
    print('Development set Accuracy: ' + str(round(bestAccuracy, 2)) + '%')

    #Reading the test set.
    fileName = 'Dataset/phishing.test'
    Xtest, Ytest = readData(fileName, maxNoOfFeatures)
    testAccuracy = predict(Xtest, Ytest, bestWeight, bestBias)
    print('Test set Accuracy: ' + str(round(testAccuracy, 2)) + '%')


if __name__ == '__main__':
    main()

