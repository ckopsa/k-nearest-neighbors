import pandas as pd
import numpy as np
def calc_entropy(*probabilities):
    return sum([0 if p == 0 else -p * np.log2(p) for p in probabilities])
def nominalize_dataframe(df, numBins, labels):
    tempdataframe = df.copy(True)
    for column in list(tempdataframe)[:-1]:
        tempdataframe[column] = nominalize_column(df[column], numBins, labels)
    return tempdataframe
def nominalize_column(column, numBins, labels):
    if not isinstance(column, str):
        return pd.cut(column, create_bins(column, numBins), labels=labels)
def create_bins(column, numBins):
    columnMin = float(min(column))
    columnMin = columnMin - columnMin/10
    columnMax = float(max(column))
    return np.linspace(columnMin, columnMax, numBins)
def getColumnEntropy(df, column_name):
    column_entropy_sum = 0
    for feature in set(df[column_name]):
        column_entropy_sum += getFeatureEntropy(df, column_name, feature)
    return column_entropy_sum
def getFeatureEntropy(df, column_name, feature):
    counts = df[df[column_name] == feature].ix[:, -1].value_counts()
    entropy = calc_entropy(*list(counts / sum(counts)))
    ratio = list(df[column_name]).count(feature) / float(df[column_name].count())
    return entropy * ratio
def getBestColumn(df):
    return list(df)[np.argmin(map(lambda(x): getColumnEntropy(df, x), list(df)[:-1]))]
def buildTree(df):
    best_column = getBestColumn(df)
    d_tree = dict()
    for feature in set(df[best_column]):
        if len(list(df)) == 2 or 0.0 == getFeatureEntropy(df, best_column, feature):
            d_tree[best_column, feature] = list(df[df[best_column] == feature].ix[:, -1])[0]
        else:
            d_tree[best_column, feature] = buildTree(df[df[best_column] == feature].drop([best_column], axis=1))
    return d_tree
def classifyDataFrame(df, d_tree):
    return [classifyRow(row, d_tree) for index, row in df.iterrows()]
def classifyRow(row, d_tree):
    current_column = d_tree.keys()[0][0]
    classification = d_tree.get((current_column, row[current_column]), d_tree[d_tree.keys()[0][0], d_tree.keys()[0][1]])
    if not isinstance(classification, dict):
        return classification
    else:
        return classifyRow(row, classification)
def accuracy(output, target):
    truePositive = 0
    falsePositive = 0
    for i in range(len(output)):
        if output[i] == target[i]:
            truePositive = truePositive + 1
        else:
            falsePositive = falsePositive + 1
    return float(truePositive) / len(output)

def displayDTree(tree):
    for key in tree.keys():
        if isinstance(tree[key], dict):
            print "IF", key[0], "==", key[1], "AND"
            displayDTree(tree[key])
        else:
            print key[0], "==", key[1]
            print "  ", tree[key]

df0 = pd.read_csv("iris.data", header = None, names = ["sepal_length",
                                                      "sepal_width",
                                                      "petal_length",
                                                      "petal_width", "class"])
names = ["very_very_small", "very_small", "small", "medium", "large", "very-large"]
nomdf0 = nominalize_dataframe(df0, len(names)+1, names)
# Get Columns Ratios
d_tree0 = buildTree(nomdf0.sample(frac=.5))
test_data0 = nomdf0.sample(frac=.2)
output0 = classifyDataFrame(test_data0, d_tree0)
target0 = list(test_data0["class"])
print accuracy(output0, target0)

df1 = pd.read_csv("lenses.data", sep='  ', names=["one", "two", "three", "four", "five"])
d_tree1 = buildTree(df1.sample(frac=.2))
test_data1 = df1.sample(frac=.8)
output1 = classifyDataFrame(test_data1, d_tree1)
target1 = list(test_data1.ix[:, -1])
print accuracy(output1, target1)

df2 = pd.read_csv("votes.data", names= ["Class Name",
                                        "handicapped-infants", "water-project-cost-sharing",
                                        "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
                                        "religious-groups-in-schools", "anti-satellite-test-ban",
                                        "aid-to-nicaraguan-contras", "mx-missile", "immigration",
                                        "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue",
                                        "crime", "duty-free-exports", "export-administration-act-south-africa"])
df2 = df2[list(df2)[::-1]]
d_tree2 = buildTree(df2.sample(frac=.5))
test_data2 = df2.sample(frac=.8)
output2 = classifyDataFrame(test_data2, d_tree2)
target2 = list(test_data2.ix[:, -1])
print accuracy(output2, target2)
displayDTree(d_tree2)
