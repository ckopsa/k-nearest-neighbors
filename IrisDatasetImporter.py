import csv
class IrisDataImporter(object):
   def __init__(self, filename):
      self.data = [row for row in csv.reader(open(filename, 'rb'))][:150]
      self.classes = list(set([row[4] for row in self.data]))
      self.targetVector = map(lambda x: self.classes.index(x), [row[4] for row in self.data])
      for x in range(0, len(data)):
         del self.data[x][4]
         self.data[x] = map(float, self.data[x])
