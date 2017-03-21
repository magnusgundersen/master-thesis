__author__ = 'magnus'
from sklearn import svm as svm
from sklearn.linear_model import ridge as ridge
from sklearn import neighbors as neig
from sklearn import neural_network
from sklearn import ensemble as ensnb
from sklearn import linear_model
from sklearn import multiclass as mclass
import time

from reservoircomputing import rc_interface as interfaces


class SVM(interfaces.RCClassifier):
    def __init__(self):
        super(SVM, self).__init__()
        self.svm = svm.SVC(kernel="linear")
        n_estimators = 10
        #self.svm = mclass.OneVsRestClassifier(ensnb.BaggingClassifier(svm.SVC(kernel='linear', probability=True, class_weight='balanced'), max_samples=1.0 / n_estimators, n_estimators=n_estimators))

    def fit(self, training_input, correct_predictions):
        fitted = self.svm.fit(training_input, correct_predictions)
        return fitted

    def predict(self, reservoir_outputs):
        predictions = self.svm.predict(reservoir_outputs)
        return predictions

class SGD(interfaces.RCClassifier):
    def __init__(self):
        super(SGD, self).__init__()
        self.sgd = linear_model.SGDClassifier(loss="perceptron")

        #self.sgd = linear_model.SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
        # SGDClassifier(loss="log", shuffle=True, penalty="l2")
        # SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None).


    def fit(self, training_input, correct_predictions):
        fitted = self.sgd.fit(training_input, correct_predictions)
        return fitted

    def predict(self, reservoir_outputs):
        predictions = self.sgd.predict(reservoir_outputs)
        return predictions

class LinReg(interfaces.RCClassifier):
    def __init__(self):
        super(LinReg, self).__init__()
        self.linreg = linear_model.LinearRegression()

    def fit(self, training_input, correct_predictions):
        fitted = self.linreg.fit(training_input, correct_predictions)
        return fitted

    def predict(self, reservoir_outputs):
        predictions = self.linreg.predict(reservoir_outputs)
        return predictions

class RandomForest(interfaces.RCClassifier):
    def __init__(self):
        super(RandomForest, self).__init__()
        self.rndForst = ensnb.RandomForestClassifier(min_samples_leaf=2)

    def fit(self, training_input, correct_predictions):
        fitted = self.rndForst.fit(training_input, correct_predictions)
        return fitted

    def predict(self, reservoir_outputs):
        predictions = self.rndForst.predict(reservoir_outputs)
        return predictions

