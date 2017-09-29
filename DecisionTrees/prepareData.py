import numpy as np
import os.path

#Returns the data for features from 1-6 only.
def getData(inFileName):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "./Updated_Dataset/" + inFileName)
    trainingFile = open(path, "r", encoding="utf8")

    Y = np.empty((0,1), int)
    X = np.empty((0,6),int)
    for line in trainingFile:
        newLine = line.strip()
        if len(newLine) == 0:
            continue
        if newLine[0] == '+':
            Y = np.append(Y, np.array([1]))
        elif newLine[0] == '-':
            Y = np.append(Y, np.array([0]))
        else:
            print("Problem with the data")
            continue

        name = newLine[2:]
        nameList = name.split(" ")
        if len(nameList) < 2:
            print("Data problematic with name")
            continue

        # Feature 1.
        x1 = 0
        if len(nameList[0]) >= len(nameList[-1]):
            x1 = 1

        #Feature 2.
        x2 = 0
        if len(nameList) >= 3:
            x2 = 1

        #Feature 3
        x3 = 0
        firstName = nameList[0]
        if firstName[0].lower() == firstName[-1].lower():
            x3 = 1

        #Feature 4
        x4 = 0
        if nameList[0] < nameList[-1]:
            x4 = 1

        #Feature 5
        x5 = 0;
        firstName = nameList[0]
        if len(firstName) >= 2:
            if (firstName[1].lower() == "a" or firstName[1].lower() == "e" or firstName[1].lower() == "i" or firstName[1].lower() == "o" or firstName[1].lower() == "u"):
                x5 = 1;

        #Feature 6
        x6 = 0
        lastName = nameList[-1]
        if len(lastName)%2 == 0:
            x6 = 1;

        X = np.vstack((X, np.array([x1,x2,x3,x4,x5,x6])))
    #print(X)
    return X,Y

#Returns data with 20 faetures.
def getMoreData(inFileName):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "./Updated_Dataset/" + inFileName)
    trainingFile = open(path, "r", encoding="utf8")

    Y = np.empty((0,1), int)
    X = np.empty((0,20),int)
    for line in trainingFile:
        newLine = line.strip()
        if len(newLine) == 0:
            continue
        if newLine[0] == '+':
            Y = np.append(Y, np.array([1]))
        elif newLine[0] == '-':
            Y = np.append(Y, np.array([0]))
        else:
            print("Problem with the data")
            continue

        name = newLine[2:]
        nameList = name.split(" ")
        if len(nameList) < 2:
            print("Data problematic with name")
            continue

        firstName = nameList[0]
        lastName = nameList[-1]

        # Feature 1.
        x1 = 0
        if len(firstName) >= len(lastName):
            x1 = 1

        #Feature 2.
        x2 = 0
        if len(nameList) >= 3:
            x2 = 1

        #Feature 3
        x3 = 0
        if firstName[0].lower() == firstName[-1].lower():
            x3 = 1

        #Feature 4
        x4 = 0
        if nameList[0] < nameList[-1]:
            x4 = 1

        #Feature 5
        x5 = 0;
        if len(firstName) >= 2:
            if (firstName[1].lower() == "a" or firstName[1].lower() == "e" or firstName[1].lower() == "i" or firstName[1].lower() == "o" or firstName[1].lower() == "u"):
                x5 = 1;

        #Feature 6
        x6 = 0
        if len(lastName)%2 == 0:
            x6 = 1;

        #Feature 7
        x7 = 0;
        if len(firstName)%2 == 0:
            x7 = 1

        #Feature 8
        x8 = 0
        if len(name) > 10:
            x8 = 1

        #Feature 9
        x9 = 0
        if firstName[0].lower() == "r":
            x9 = 1

        #Feature 10
        x10 = 0
        if firstName[0].lower() == "a":
            x10 = 1

        #Feature 11
        x11 = 1
        if len(nameList) > 3:
            x11 = 0

        #Feature 12
        x12 = 0
        if firstName[0].lower() == "m":
            x12 = 1

        #Feature 13
        x13 = 1
        if len(nameList) >= 3:
            if "." in nameList[1]:
                x13 = 0

        #Feature 14
        x14 = 1
        if lastName[0].lower() == "r":
            x14 = 0

        #Feature 15
        x15 = 0
        if firstName[0].lower() == "d":
            x15 = 1

        #Feature 16
        x16 = 0
        if lastName[0].lower() == "t":
            x16 = 1

        #Feature 17
        x17 = 0
        if firstName[0].lower() == "p":
            x17 = 1

        #Feature 18
        x18 = 0
        if firstName[0].lower() == "j":
            x18 = 1

        #Feature 19
        x19 = 1
        if len(nameList) == 3:
            x19 = 0

        #Feature 20
        x20 = 0
        if lastName[0].lower() == "W":
            x20 = 1

        X = np.vstack((X, np.array([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20])))
    #print(X)
    return X,Y