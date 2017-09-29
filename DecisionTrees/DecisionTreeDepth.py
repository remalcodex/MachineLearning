import numpy as np
from prepareData import getMoreData

maxFeatureValues = [2,2,2,2,2,2]
majorityLabel = -1

#This function calculates the entropy of the set.
def Entropy(inSet):
    entropy = 0
    labels, counts = np.unique(inSet, return_counts=True)
    probabilites = counts.astype('float')/len(inSet)
    for p in probabilites:
        if p != 0.0:
            entropy -= p * np.log2(p)
    return entropy


# This function returns the index of feature with maximum information gain.
def information_gain(inX, inY, inVisitedFeatures):
    gainList = np.empty((0,1), float)
    #print("Printing gain list.")

    for xFeature in inX.T:
        gain = Entropy(inY)
        labels, counts = np.unique(xFeature, return_counts=True)
        probabilities = counts.astype('float')/len(xFeature)
        xSet = zip(probabilities, labels)

        for p, label in xSet:
            entropyT = Entropy(inY[xFeature == label])
            gain -= p*entropyT
        gainList = np.append(gainList, gain)

    #Calculating max gain.
    maxGain = -1
    maxIndex = -1
    counter = 0
    while counter < len(inVisitedFeatures):
        # We skip those features that we have already visited and are a part of the tree.
        # inVisitedFeatures[x] = 1 means the feature has been visited.
        if inVisitedFeatures[counter] != 1:
            if maxGain < gainList[counter]:
                maxGain = gainList[counter]
                maxIndex = counter
        counter = counter + 1

    return maxIndex, maxGain


#This function partitions the set and returns the dictionary with unique values.
def split(inXFeature):
    dict = {}
    labels = np.unique(inXFeature)
    for label in labels:
        val = (inXFeature == label).nonzero()
        dict[label] = val[0] #Adding [0] here since the nonzero function in previous line give back values in row major order and there is one empty row always.

    return dict


# Code to build the tree.
def buildTree(inX, inY, inVisitedFeatures, depth):

    #Take majority of inY when depth is -1
    if depth == -1:
        labels, counts = np.unique(inY, return_counts=True)
        return labels[np.argmax(counts)]

    depth -= 1

    if len(inY) == 0:
        return inY

    # Checking if the input Y is leaf node or not.
    isLeaf = "true"
    val0 = inY[0]
    for val in inY:
        if val0 != val:
            isLeaf = "false"
    if isLeaf == "true":
        return inY[0]

    # If all the features have been visited till now return with the majority of input labels.
    if set(inVisitedFeatures) == {1}:
        labels, counts = np.unique(inY, return_counts=True)
        return labels[np.argmax(counts)]

    xIndex, gainVal = information_gain(inX, inY, inVisitedFeatures)

    #making copy of visitedFeatures since by default function call is pass by reference.
    newVisitedFeatures = inVisitedFeatures[:]
    newVisitedFeatures[xIndex] = 1 #Marking feature as visited.

    #Splitting the data
    splitX = split(inX[:, xIndex])

    #Creating a value for the tree.
    #First position holds the feature xIndex.
    totalEntries = [xIndex]
    #Adding the number of lists equal to number of values feature can take.
    for index in range(maxFeatureValues[xIndex]):
        L = list()
        totalEntries.append(L)
    tree = totalEntries

    for index in range(maxFeatureValues[xIndex]):
        if index in splitX:
            val = splitX.get(index)
            xSubset = inX.take(val, axis=0)
            ySubset = inY.take(val, axis=0)

            #index+1 because first position is taken by the feature index.
            tree[index+1] = buildTree(xSubset, ySubset, newVisitedFeatures, depth)
        else:
            # index+1 because first position is taken by the feature index.
            tree[index+1] = majorityLabel

    return tree


#Code to traverse the tree.
def traverseTree(inX, inTree):

    if isinstance(inTree, list):
        xFeature = inTree[0]
        xFeatureVal = inX[xFeature]
        newTree = inTree[xFeatureVal+1]
        yVal = traverseTree(inX, newTree)
        return yVal
    else:
        return inTree

#This function calculates the accuracy and builds the tree.
def buildAndCalculateTree(inX, inY, cvX, cvY, depth):

    # Getting the majority label.
    labels, counts = np.unique(inY, return_counts=True)
    global majorityLabel
    majorityLabel = labels[np.argmax(counts)]

    #Setting the number of features, maxFeatureValues and visitedFeatures
    noOfFeatures = inX.shape[1]
    global maxFeatureValues
    maxFeatureValues = [2] * noOfFeatures
    global visitedFeatures
    visitedFeatures = [0] * noOfFeatures

    #This depth value will decrease with each recursion step.
    mainTree = buildTree(inX, inY,visitedFeatures, depth)
    #print(mainTree)

    #Training data accuracy.
    counter = 0
    correct = 0
    incorrect = 0
    for row in inX:
        yVal = traverseTree(row, mainTree)
        realYVal = inY[counter]
        if yVal == realYVal:
            correct = correct + 1
        else:
            incorrect = incorrect +1
        counter = 1 + counter
    trainAccuracy = correct/(incorrect+correct) * 100
    #print("Training Accuracy: " + ("%f" % trainAccuracy) + "%")

    #Cross Validation data accuracy.
    counter = 0
    correct = 0
    incorrect = 0
    for row in cvX:
        yVal = traverseTree(row, mainTree)
        realYVal = cvY[counter]
        if yVal == realYVal:
            correct = correct + 1
        else:
            incorrect = incorrect +1
        counter = 1 + counter
    cvAccuracy = correct/(incorrect+correct) * 100
    #print("Cross Validation Accuracy: " + ("%f" % cvAccuracy) + "%")

    return mainTree, trainAccuracy, cvAccuracy

#This function creates a tree from the entire training data and checks it on the test data.
def trainFinaldata(inDepth):
    X, Y = getMoreData("updated_train.txt")
    testX,testY = getMoreData("updated_test.txt")
    fTree, fTrainAccuracy, fTestAccuracy = buildAndCalculateTree(X, Y, testX, testY, inDepth)
    print("Final Training...")
    print("Depth:- " + ("%d" % inDepth))
    print(fTree)
    print("Train Accuracy: " + ("%f" % fTrainAccuracy))
    print("DecisionTrees Accuracy: " + ("%f" % fTestAccuracy))
    return


def main():
    #Getting the training data from 4 files in the splits.
    listX = []
    listY = []
    for i in range(4):
        fname = "Updated_CVSplits/updated_training0" + ("%d" % i) + ".txt"
        X, Y = getMoreData(fname)
        listX.append(X)
        listY.append(Y)

    noOfFeatures = listX[0].shape[1]

    # This part selects one depth at a time and calculates the mean and standard deviation for each depth.
    meanList = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
    counter = 0
    depthArray = [1, 2, 3, 4, 5, 10, 15, 20]
    for depth in depthArray:
        treeList = []
        trainAccuracyList = np.array([0,0,0,0])
        cvAccuracyList = np.array([0,0,0,0])

        # This part takesout 1 out of 4 training sets and calculates the tree, training accuracy and the Cross validation accuracy.
        takeOut = 0
        for i in range(4):
            doOnce = 0
            newX = np.empty((0, noOfFeatures), int)
            newY = 0
            for j in range(4):
                if (j == takeOut):
                    continue
                else:
                    if doOnce == 0:
                        newY = listY[j]
                        doOnce = 1
                    else:
                        newY = np.hstack((newY, listY[j]))
                    newX = np.vstack((newX, listX[j]))

            cvX = listX[takeOut]
            cvY = listY[takeOut]
            tree, trainAccuracy, cvAccuracy = buildAndCalculateTree(newX, newY, cvX, cvY, depth) #Calling the buliding of tree
            treeList.append(tree)
            trainAccuracyList[i] = trainAccuracy
            cvAccuracyList[i] = cvAccuracy
            takeOut += 1 #Take out the next training set from training.

        #Calculating mean and standard deviation.
        mean = np.sum(cvAccuracyList)/4
        stdDeviation = cvAccuracyList - mean
        stdDeviation = stdDeviation * stdDeviation
        stdDeviation = sum(stdDeviation)
        stdDeviation = stdDeviation / 4
        stdDeviation = np.sqrt(stdDeviation)
        meanList[counter] = mean
        counter += 1
        print("Depth:- " + ("%d" % depth))
        print("Mean: " + ("%f" % mean))
        print("Deviation: " + ("%f" % stdDeviation))

    print("\n\n-------------------------------")
    #Calculating the optimal depth from the list of means.
    index = np.argmax(meanList);
    optimalDepth = depthArray[index]
    print("OptimalDepth: " + ("%d" % optimalDepth))

    #Now training on the final test set with optimal depth.
    trainFinaldata(optimalDepth)

# Main script.
if __name__ == '__main__':
    main()

