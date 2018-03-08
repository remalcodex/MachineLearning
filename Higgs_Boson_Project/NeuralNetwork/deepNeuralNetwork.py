import tensorflow as tf
import numpy as np
from newBatch import Dataset

def getNextBatch(X, Y):
    dataset = Dataset(np.arange(0, 10))
    for i in range(10):
        print(dataset.next_batch(5))

def createNeuralNetworModel(inData):
    noNodesHL1 = 500
    noNodesHL2 = 500
    noNodesHL3 = 500

    noClasses = 2

    noOfFeatures = 28

    hiddenLayer1 = {'weights': tf.Variable(tf.random_normal([noOfFeatures, noNodesHL1])),
                    'biases': tf.Variable(tf.random_normal([noNodesHL1]))}

    hiddenLayer2 = {'weights': tf.Variable(tf.random_normal([noNodesHL1, noNodesHL2])),
                    'biases': tf.Variable(tf.random_normal([noNodesHL2]))}

    hiddenLayer3 = {'weights': tf.Variable(tf.random_normal([noNodesHL2, noNodesHL3])),
                    'biases': tf.Variable(tf.random_normal([noNodesHL3]))}

    outpuLayer = {'weights': tf.Variable(tf.random_normal([noNodesHL3, noClasses])),
                  'biases': tf.Variable(tf.random_normal([noClasses]))}

    #Fix the data here.
    #Maybe use + instead of tf.add, just loke in the video.
    l1 = tf.add(tf.matmul(inData, hiddenLayer1['weights']), hiddenLayer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hiddenLayer2['weights']), hiddenLayer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hiddenLayer3['weights']), hiddenLayer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, outpuLayer['weights']) + outpuLayer['biases']
    return output

def trainNeuralNetwork(X, Y, Xtest, Ytest):
    Y = np.where(Y > 0.5, [1, 0], [0, 1])
    Ytest = np.where(Ytest > 0.5, [1, 0], [0, 1])

    batchSize = 200
    noOfFeatures = 28
    noOfExamples = X.shape[0]

    xTF = tf.placeholder('float', [None, noOfFeatures])
    yTF = tf.placeholder('float')

    prediction = createNeuralNetworModel(xTF)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=yTF))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 15
    #config = tf.ConfigProto(log_device_placement=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            epochLoss = 0
            dataset = Dataset(X, Y)
            for i in range(int(noOfExamples/batchSize)):
                epoch_x, epoch_y = dataset.next_batch(batchSize)
                #epoch_x, epoch_y = mnist.train.next_batch(batchSize)
                _, c = sess.run([optimizer, cost], feed_dict={xTF: epoch_x, yTF: epoch_y})
                epochLoss += c

            print('Epoch', epoch, 'completed out of', epochs,'loss:', epochLoss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(yTF, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        dataset = Dataset(Xtest, Ytest)
        batchSizeTest = 100
        accuracyVal = 0
        for i in range(int(Xtest.shape[0] / batchSizeTest)):
            epoch_x, epoch_y = dataset.next_batch(batchSizeTest)
            accuracyVal += accuracy.eval({xTF: epoch_x, yTF: epoch_y})
        print('Accuracy:', accuracyVal/batchSizeTest)
