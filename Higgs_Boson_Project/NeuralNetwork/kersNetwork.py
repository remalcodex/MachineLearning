import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import MaxPooling1D
import numpy as np
from newBatch import Dataset
from sklearn.metrics import confusion_matrix

def trainKerasNetwork(X, Y, Xtest, Ytest):
    # Y = np.where(Y > 0.5, [1, 0], [0, 1])
    # Ytest = np.where(Ytest > 0.5, [1, 0], [0, 1])

    batchSize = 200
    noOfFeatures = 28
    noOfExamples = X.shape[0]

    #-------------------------------------------------------------------------
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim=28, init='uniform', activation=None, input_dim=noOfFeatures))
    classifier.add(Dense(output_dim=56, init='uniform', activation=None))
    #classifier.add(Dense(output_dim=28, init='uniform', activation='relu'))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim=28, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=56, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=28, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=14, init='uniform', activation='relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim=7, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    dataset_test = Dataset(Xtest, Ytest)
    batchSizeTest = 100
    accuracyVal = 0

    classifier.fit(X, Y, batch_size=batchSize, nb_epoch=100)
    y_pred = classifier.predict(Xtest, batch_size=batchSize)
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(Ytest, y_pred)
    print(cm)
