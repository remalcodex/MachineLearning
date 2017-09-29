import numpy as np
from prepareData import getData

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
def buildTree(inX, inY, inVisitedFeatures):

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

    #Calling the recursion equal to number of feature values
    for index in range(maxFeatureValues[xIndex]):
        if index in splitX:
            val = splitX.get(index)
            xSubset = inX.take(val, axis=0) # Takes the value fromthe np array inX for all indices present in val.
            ySubset = inY.take(val, axis=0)

            # index+1 because first position is taken by the feature index.
            tree[index+1] = buildTree(xSubset, ySubset, newVisitedFeatures)
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

# Main script.
def main():
    #Getting the data.
    X, Y = getData("updated_train.txt")

    # Getting the majority label.
    labels, counts = np.unique(Y, return_counts=True)
    global majorityLabel
    majorityLabel = labels[np.argmax(counts)]

    #Setting the number of features, maxFeatureValues and visitedFeatures
    noOfFeatures = X.shape[1]
    global maxFeatureValues
    maxFeatureValues = [2] * noOfFeatures
    global visitedFeatures
    visitedFeatures = [0] * noOfFeatures

    mainTree = buildTree(X,Y,visitedFeatures)
    print(mainTree)

    #Training data accuracy.
    counter = 0
    correct = 0
    incorrect = 0
    for row in X:
        yVal = traverseTree(row, mainTree)
        realYVal = Y[counter]
        if yVal == realYVal:
            correct = correct + 1
        else:
            incorrect = incorrect +1
        counter = 1 + counter
    print("Training Accuracy: " + ("%f" % (correct/(incorrect+correct) * 100)) + "%")

    #DecisionTrees data accuracy.
    counter = 0
    correct = 0
    incorrect = 0
    newX, newY = getData("updated_test.txt")
    for row in newX:
        yVal = traverseTree(row, mainTree)
        realYVal = newY[counter]
        if yVal == realYVal:
            correct = correct + 1
        else:
            incorrect = incorrect +1
        counter = 1 + counter
    print("DecisionTrees Accuracy: " + ("%f" % (correct/(incorrect+correct) * 100)) + "%")


if __name__ == '__main__':
    main()

