__author__ = 'magnus'
from sklearn import svm as svm
from sklearn.linear_model import ridge as ridge
from sklearn import neighbors as neig
from sklearn import neural_network
from sklearn import linear_model
import time

from reservoircomputing import rc_interface as interfaces


class SVM(interfaces.RCClassifier):
    def __init__(self):
        super(SVM, self).__init__()
        self.svm = svm.SVC(kernel="linear")
        #self.svm = ridge.Ridge()
        #self.svm = neig.KNeighborsClassifier()
        #self.svm = neural_network.MLPClassifier(hidden_layer_sizes=())
        #self.svm = linear_model.LinearRegression()
        self.svm = linear_model.SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
        # SGDClassifier(loss="log", shuffle=True, penalty="l2")
        # SGDClassifier(loss=”perceptron”, eta0=1, learning_rate=”constant”, penalty=None).

    def fit(self, training_input, correct_predictions):
        #print(str(len(training_input)))
        #self.prev_seen = training_input
        before = time.time()
        fitted = self.svm.fit(training_input, correct_predictions)
        after = time.time()
        return fitted

    def predict(self, reservoir_outputs):


        #print("PREdicting:")
        #print(reservoir_outputs)
        predictions = self.svm.predict(reservoir_outputs)

        #print(predictions)
        #print("")

        if False: #eservoir_outputs in self.prev_seen:
            chuncks = len(reservoir_outputs)//5
            print(len(reservoir_outputs)/5)
            print("seen before!" + str(predictions))
            for i in range(5):
                print(reservoir_outputs[i*chuncks:(i+1)*chuncks])
            print("")
        return predictions
