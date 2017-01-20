class HardCodedClassifier(object):
    def __init__(self):
        pass
    def fit(self, inputVector, targetVector):
        pass
    def predict(self, inputVector):
        return [self.classify(x) for x in range(len(inputVector))]
    def classify(self, instance):
        return 0
