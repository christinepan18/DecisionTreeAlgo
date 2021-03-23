import numpy as np
from pandas.core.indexes import base
import scipy
from scipy.stats import mode, entropy
import pandas as pd
from pprint import pprint
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

def intInfo(a: np.ndarray):
    return np.sum(-np.log2(a/a.size)*a/a.size)

def giniIndex(a: np.ndarray):
    return 1 - np.sum(a**2)

class Node:
    def __init__(self, label, isLeaf=False, minSampleLeaf=1):
        self.label = label
        self.isLeaf = isLeaf
        self.children = []
        self.minSampleLeaf = minSampleLeaf
        self.attrIdx = None
        self.val = None
        self.type = None # 'cont' or 'cat'
        self.info = None

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __str__(self) -> str:
        return str(self.__dict__)

    def _train(self, X, y: np.ndarray, properties, propIndex, method='entropy'):
        """
        X: attrs
        y: classes
        properties: list of propaerties of each attr
            ['cat', 'cont', 'cat', 'cont']
        """
        gainFunc = self._gainFunc(method)
        #print(properties)
        assert X.shape[0] == y.size
        classes, counts = np.unique(y, return_counts=True)
        # Apply root to be labelled as the most popular class
        mostPopular = mode(y)[0][0]
        if len(y) <= self.minSampleLeaf or classes.size == 1 or len(properties) <= 0: return Node(mostPopular, isLeaf=True)
        I = gainFunc(counts/y.size)
        self.info = I
        """
        For each column, calculate the entropy of the column of data
        """
        gain = 0
        bestAttribute = None
        knownValues = None
        for i in range(X.shape[1]):
            # calculate gain
            col = X[:, i]
            possibleKV = None
            if properties[i] == 'cat':
                newGain, possibleKV = self._information_cat(col, y, I, method)
            elif properties[i] == 'cont':
                colSort = np.sort(np.unique(col))
                # One single unique value, cannot split
                if colSort.size <= 1:
                    continue
                newGain, possibleKV = self._informationCont(col, y, I, method)
            if newGain >= gain:
                bestAttribute = i
                knownValues = possibleKV
                gain = newGain
        self.type = properties[bestAttribute]
        self.attrIdx = propIndex[bestAttribute]
        if self.type == 'cat':
            rootValue = []
            for val in knownValues:
                rootValue.append(val)
                xSub, ySub = X[X[:, bestAttribute] == val], y[X[:, bestAttribute] == val]
                if ySub.size <= 0:
                    self.children.append(Node(mostPopular, isLeaf=True))
                else:
                    self.children.append(Node(None)._train(
                                        np.delete(xSub, bestAttribute, axis=1),
                                        ySub,
                                        properties=properties[:bestAttribute] + properties[bestAttribute+1:],
                                        propIndex=propIndex[:bestAttribute] + propIndex[bestAttribute+1:],
                                        method=method))
            self.val = rootValue
        elif self.type == 'cont':
            val = knownValues[0] # single Y/N split
            self.val = val
            xValueH, yValueH = X[X[:, bestAttribute] > val], y[X[:, bestAttribute] > val]
            xValueL, yValueL = X[X[:, bestAttribute] <= val], y[X[:, bestAttribute] <= val]
            if yValueL.size <= 0:
                self.children.append(Node(mostPopular, isLeaf=True))
            else:
                self.children.append(Node(None)._train(
                                        np.delete(xValueL, bestAttribute, axis=1),
                                        yValueL,
                                        properties=properties[:bestAttribute] + properties[bestAttribute+1:],
                                        propIndex=propIndex[:bestAttribute] + propIndex[bestAttribute+1:],
                                        method=method))
            if yValueH.size <= 0:
                self.children.append(Node(mostPopular, isLeaf=True))
            else:
                self.children.append(Node(None)._train(
                                        np.delete(xValueH, bestAttribute, axis=1),
                                        yValueH,
                                        properties=properties[:bestAttribute] + properties[bestAttribute+1:],
                                        propIndex=propIndex[:bestAttribute] + propIndex[bestAttribute+1:],
                                        method=method))
        return self

    def _informationCont(self, col, y, info, method='entropy'):
        gainFunc = self._gainFunc(method)
        colSort = np.sort(np.unique(col))
        splitPoints = (colSort[1::] + colSort[0:-1]) * 0.5
        possibleIRes = np.zeros_like(splitPoints)
        index = 0
        for sp in splitPoints:
            yHigh, yLow = y[col > sp], y[col <= sp]
            for D in [yHigh, yLow]:
                __, condCounts = np.unique(D, return_counts=True)
                condClassEntropy = gainFunc(condCounts/D.size)
                possibleIRes[index] += D.size/ y.size * condClassEntropy
            index += 1
        iResIndex= np.argmin(possibleIRes)
        iRes = possibleIRes[iResIndex]
        bestSplit = splitPoints[iResIndex]
        possibleKV = np.array([bestSplit])
        valCounts = np.array([np.sum(col > bestSplit), np.sum(col <= bestSplit)])
        if method == "gain_ratio":
            gain = (info - iRes) / intInfo(valCounts/y.size)
        else: gain = info - iRes
        return gain, possibleKV

    def _information_cat(self, col, y, info, method='entropy'):
        gainFunc = self._gainFunc(method)
        iRes = 0
        possibleKV, valCounts = np.unique(col, return_counts=True)
        for val, val_count in zip(possibleKV, valCounts):
            y_val = y[col == val]
            __, condCounts = np.unique(y_val, return_counts=True)
            condClassEntropy = gainFunc(condCounts/y_val.size)
            iRes += val_count/ y.size * condClassEntropy
        if method == "gain_ratio":
            gain = (info - iRes) / intInfo(valCounts/y.size)
        else: gain = info - iRes
        return gain, possibleKV

    def _gainFunc(self, method):
        if method == 'gini':
            gainFunc = lambda p: giniIndex(p)
        else:
            gainFunc = lambda p: entropy(p, base=2)
        return gainFunc

    @classmethod
    def train(cls, X, y, properties, method='entropy'):
        return Node(None)._train(X, y, properties, list(range(len(properties))), method=method)


class DecisionTreeClassifier:

    def __init__(self, minSampleLeaf=1) -> None:
        self.minSampleLeaf = minSampleLeaf
        self.root = Node(None)


    def train(self, X, y, properties=[], method='entropy'):
        if not properties: properties = self._attr_types(X)
        self.root = Node.train(X, y, properties, method=method)

    def _attr_types(self, X, catThresh=3):
        properties = []
        for i in range(X.shape[1]):
            if np.unique(X[:, i]).size > catThresh:
                properties.append('cont')
            else: properties.append('cat')
        return properties

    def predict(self, X):

        if X.ndim < 2:
            xMat = np.array([X])
        else: xMat = np.array(X)
        assert xMat.ndim == 2
        pred = np.array([])
        # print(xMat)
        for i in range(xMat.shape[0]):
            vec = xMat[i]
            # print(vec)
            label = DecisionTreeClassifier._traverse(vec, self.root)
            pred = np.append(pred, label)
        return pred

    @classmethod
    def _traverse(cls, vec, r: Node):
        if r.isLeaf:
            return r.label
        else:
            bestValue = vec[r.attrIdx]
            if r.type == 'cont':
                if bestValue <= r.val:
                    return DecisionTreeClassifier._traverse(vec, r.children[0])
                else:
                    return DecisionTreeClassifier._traverse(vec, r.children[1])
            else:
                #print(r.val)
                idx = r.val.index(bestValue)
                return DecisionTreeClassifier._traverse(vec, r=r.children[idx])

def formatTree(t,s,labels=["outlook", "temp", "humidity", "windy"]):
    if not t.isLeaf:
        print("----"*s+ "attr: {}, vals: {}, info: {}".format(labels[t.attrIdx], t.val, t.info))
    else:
        print("----"*s + "label: {}, vals: {}, idx: {}".format(t.label, t.val, t.attrIdx))
    for child in t.children:
        formatTree(child,s+1,labels=labels)

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/panpa/OneDrive/Desktop/tennisdata.csv")
    X = df.drop("play",axis=1).to_numpy()
    Y = df.play.to_numpy()
    testValue = pd.DataFrame(
        {
            "outlook": ["overcast"],
            "temp": [60],
            "humidity": [62],
            "windy": [False]
        }
    )
    df = df.append(testValue, ignore_index=True)
    node = Node.train(X, Y, ["cat", "cont", 'cont', 'cat'])
    #print(node.children[0])
    #formatTree(node, 0)
    
    transData = pd.get_dummies(df[['outlook', 'temp', 'humidity', 'windy']], drop_first=True)
    baseline = DTC(criterion="gini")
    baseline.fit(transData.iloc[:-1], Y)
    print("Baseline prediction: ", baseline.predict(transData.iloc[-1].to_numpy().reshape(1, -1)))
    model = DecisionTreeClassifier()
    model.train(transData.iloc[:-1].to_numpy(), Y, properties=["cont", "cont", 'cont', 'cont', 'cont'], method='gini')
    formatTree(model.root, 0, labels=transData.columns.tolist())
    print("Impl predict: ", model.predict(transData.iloc[-1]))
    plot_tree(baseline, feature_names=transData.columns)
    plt.show()
