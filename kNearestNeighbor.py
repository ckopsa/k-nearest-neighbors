from sklearn import datasets
from classifier import Classifier
from random import randrange

# http://rosettacode.org/wiki/Knuth_shuffle#Python
def knuth_shuffle(x, y):
    for i in range(len(x)-1, 0, -1):
        j = randrange(i + 1)
        x[i], x[j] = x[j], x[i]
        y[i], y[j] = y[j], y[i]

def accuracy(output, target):
    truePositive = 0
    falsePositive = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            truePositive = truePositive + 1
        else:
            falsePositive = falsePositive + 1
    return float(truePositive) / len(output)
def run():
    iris = datasets.load_iris()
    inputVector = iris.data
    targetVector = iris.target

    knuth_shuffle(inputVector, targetVector)

    # Split the data-set into a training set and a testing set
    trainInput = inputVector[:100:]
    trainTarget = targetVector[:100:]
    testInput = inputVector[100::]
    testTarget = targetVector[100::]

    classifier = Classifier()
    classifier.train(trainInput, trainTarget)
    testOutput = classifier.predict(testInput)

    print accuracy(testOutput, testTarget)
run()
