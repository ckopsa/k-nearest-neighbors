import numpy
class KopsaClassifier(object):
    def fit(self, inputvector, targetvector):
        self.inputvector = inputvector
        self.targetvector = targetvector
    def knn(self, instance, k):
        # find distance from instance for each element in inputvector
        distances = ((self.inputvector - instance)**2).sum(axis=1)
        # sort distances
        indices = numpy.argsort(distances, axis=0)
        nearestNeighbors = [self.targetvector[i] for i in indices[:k]]
        return nearestNeighbors
    def predict(self, inputvector):
        return [self.classify(x, 5) for x in inputvector]
    def classify(self, instance, k):
        nearestNeighbors = self.knn(instance, 5)
        return max(set(nearestNeighbors), key=nearestNeighbors.count)
