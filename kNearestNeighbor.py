from sklearn import datasets
from sklearn import model_selection
from HardCodedClassifier import HardCodedClassifier
from KopsaClassifier import KopsaClassifier


def accuracy(output, target):
    truePositive = 0
    falsePositive = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            truePositive = truePositive + 1
        else:
            falsePositive = falsePositive + 1
    return float(truePositive) / len(output)

def run(inputVector, targetVector):
    # Shuffle input and target
    # knuth_shuffle(inputVector, targetVector)
    trainInput, testInput, trainTarget, testTarget = model_selection.train_test_split(inputVector,
                                                                      targetVector,
                                                                      test_size=0.33)
    classifier = KopsaClassifier()
    classifier.fit(trainInput, trainTarget)
    testOutput = classifier.predict(testInput)

    print accuracy(testOutput, testTarget)

iris = datasets.load_iris()
run(iris.data, iris.target)
