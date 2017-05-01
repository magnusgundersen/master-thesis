"""
ReCA-system that is implemented.

"""
__author__ = 'magnus'
from classifier import skl_clfs as scikit_clfs
from classifier import tfl_ann as tflann
from reservoir import ca as ca
from reservoircomputing import rc as rc
from reservoircomputing import rc_interface as rc_if
from reca import encoder as enc
from reca import time_transition_function as time_trans
import random
import time
import pprint

class ReCASystem:
    """
    Sets up the CA
    Set the CA as a reservoir
    etc...
    """
    def __init__(self):
        self.reservoir = None
        self.classifier = None
        self.encoder = None
        self.rc_framework = None

        self.reCA_config = None

    # Below is experimental code
    def set_problem(self, reCA_problem):
        self.reCA_problem = reCA_problem

    def set_config(self, reCA_config):
        self.reCA_config = reCA_config

    def initialize_rc(self):
        self.reCA_config.encoder.create_mappings(self.reCA_problem.get_input_size())
        rc_helper = rc.RCHelper(self.reCA_config)
        self.rc_framework = rc.ReservoirComputingFramework()
        self.rc_framework.set_helper(rc_helper)

    def tackle_ReCA_problem(self):
        # 1. Sequential or not??
        # 2. Seq to seq? Or sequential classification?
        # All problems at this stage must be binary!
        if self.reCA_problem is None:
            raise ValueError("No ReCAProblem set!")

        self.example_data = None

        training_data = self.reCA_problem.training_data
        #pprint.pprint(training_data)
        #print("training data size:" + str(len(training_data)) + "shapes:" + str(training_data[0].shape))

        # For each input.
        for training_example in training_data:
            output = self.rc_framework.fit_to_data(training_example)  # Only one timestep
            if self.example_data is None:
                self.example_data = output

        self.rc_framework.train_classifier()


    def test_semi_dynamic_sequence_data(self):
        """
                :return:

                """
        if self.reCA_problem is None:
            raise ValueError("No reCAProblem set!")

        reCA_output = ReCAOutput()
        reCA_output.reCA_config = self.reCA_config

        # divide testing data:
        test_data = self.reCA_problem.testing_data
        reCA_output.all_test_examples = test_data

        number_of_correct = 0

        for test_ex in test_data:
            #  We now have a timeseries of data, on which the rc-framework must be fitted
            input_X = test_ex[0]
            output_Y = test_ex[1]
            outputs = self.rc_framework.predict(input_X)
            reCA_output.all_RCOutputs.append(outputs)
            pointer = 0
            all_correct = True
            predictions = []
            for output in output_Y:
                predictions.append(outputs[pointer])
                if output != outputs[pointer]:
                    #print("WRONG: " + str(output) + str( "  ") + str(outputs[pointer]))
                    all_correct = False
                pointer += 1
            reCA_output.all_predictions.append(predictions)

            if all_correct:
                number_of_correct += 1

        # print("Number of correct: " + str(number_of_correct) +" of " + str(len(test_data)))

        reCA_output.total_correct = number_of_correct
        return reCA_output

    def test_fully_dynamic_sequence_data(self):
        """
                :return:

                """
        if self.reCA_problem is None:
            raise ValueError("No reCAProblem set!")

        reCA_output = ReCAOutput()
        reCA_output.reCA_config = self.reCA_config

        # divide training_data:
        test_data = self.reCA_problem.testing_data
        prediction_stop_signal = self.reCA_problem.prediction_end_signal
        reCA_output.all_test_examples = test_data

        number_of_correct = 0

        for test_ex in test_data:
            #  We now have a timeseries of data, on which the rc-framework must be fitted
            input_X = test_ex[0]
            output_Y = test_ex[1]
            predicted_outputs = self.rc_framework.predict_dynamic_sequence(input_X, prediction_stop_signal)
            reCA_output.all_RCOutputs.append(predicted_outputs)
            pointer = 0
            all_correct = True
            predictions = []
            #print("Number of predicted outputs: " + str(len(predicted_outputs)))
            #print("number of correct outputs:   " + str(len(output_Y)))

            if len(predicted_outputs) > len(output_Y):  # System was not able to stop in time
                print("system did not stop in time")
                for predicted_output in predicted_outputs:
                    predictions.append(predicted_output)
                reCA_output.all_predictions.append(predictions)

            elif len(predicted_outputs) < len(output_Y):  # System stopped too soon
                print("system stopped too soon")
                for output in output_Y:
                    if pointer>=len(predicted_outputs):
                        dummy_output = "0"
                        predictions.append(dummy_output)
                    else:
                        predictions.append(predicted_outputs[pointer])
                reCA_output.all_predictions.append(predictions)

            else:  # System stopped on time
                print("system stopped on time")
                for predicted_output in predicted_outputs:
                    predictions.append(predicted_output)
                reCA_output.all_predictions.append(predictions)


        reCA_output.total_correct = 12
        return reCA_output
    def test_on_problem(self, test_set_size=0):
        """


        :return:

        """
        if self.reCA_problem is None:
            raise ValueError("No reCAProblem set!")
        reCA_output = ReCAOutput()
        reCA_output.reCA_config = self.reCA_config


        # divide training_data:
        test_data = self.reCA_problem.testing_data
        reCA_output.all_test_examples=test_data

        number_of_correct = 0

        for test_ex in test_data:
            #  We now have a timeseries of data, on which the rc-framework must be fitted
            input_X = test_ex[0]
            output_Y = test_ex[1]
            outputs = self.rc_framework.predict(input_X)
            reCA_output.all_RCOutputs.append(outputs)
            pointer = 0
            all_correct = True
            predictions = []
            for output in output_Y:
                predictions.append(outputs[pointer])
                if output != outputs[pointer]:
                    #print("WRONG: " + str(output) + str( "  ") + str(outputs[pointer]))
                    all_correct = False
                pointer += 1
            reCA_output.all_predictions.append(predictions)

            if all_correct:
                number_of_correct += 1

        # print("Number of correct: " + str(number_of_correct) +" of " + str(len(test_data)))

        reCA_output.total_correct = number_of_correct
        return reCA_output

    def get_example_run(self):
        return self.example_data

class ReCAOutput:
    def __init__(self):
        self.all_RCOutputs = []  # Includes full iterations, transitioned and input
        self.all_predictions = []
        self.correct_predictions = []
        self.all_test_examples = []
        self.reCA_config = None
        self.total_correct = 0

    def was_successful(self):
        if self.total_correct == len(self.all_test_examples):  # Success criteria
            return True
        return False




class ReCAProblem:
    """
    This class is used to precisely describe problems that may be feeded to the reca-system



    """
    def __init__(self, data_interpreter):
        """
        The example runs parameter is used to input some example data to the system. Must be on the form:

        data =
        [
        [ (input, output),
          (input, output)
        ],
        [ (input, output),
          (input, output)
        ]
        ]

        Each list in the list corresponds to a temporal run of the system. If the problem is non-temporal the following
        data set is used:

        data =
        [
        [(input, output)],
        [(input, output)]
        ]

        :param example_runs:
        """
        # Declare variables
        self.data_interpreter = None
        self.is_feed_forward = False
        self.is_fixed_sequence = False
        self.is_dynamic_sequence = False
        self.prediction_end_signal = None
        self.prediction_input_signal = None
        self.training_data = []
        self.testing_data = []
        self.input_size = 0


        self.initialize_data(data_interpreter)
    @staticmethod
    def check_data_validity(ex_data):

        try:
            training_set_size = len(ex_data)
        except Exception as e:
            raise ValueError("Data must be a list!" + str(e))


        try:
            number_of_time_steps = len(ex_data[0])
        except Exception as e:
            raise ValueError("Data in a training-example was bad " + str(e))

    def initialize_data(self, data_interpreter, testing_portion=0.0):
        # Determine if it is a feed-forward classification task
        example_data = data_interpreter.get_training_data()
        for data in example_data:
            if len(data) == 1:
                self.is_feed_forward = True

            else:
                self.is_feed_forward = False

        first_example_sequence_length = len(example_data[0][0])
        for data in example_data:
            if len(data[0]) == first_example_sequence_length:
                self.is_fixed_sequence = True

            else:
                self.is_dynamic_sequence = True
                self.is_fixed_sequence = False
                try:
                    self.prediction_end_signal = data_interpreter.get_prediction_end_signal()
                    self.prediction_input_signal = data_interpreter.get_prediction_input_signal()
                except:
                    pass
                break  # we know at least some training-examples are of different lengths

        self.testing_data = data_interpreter.get_testing_data()

        if len(self.testing_data) == 0:
            testing_portion = 0.1
            self.testing_data = example_data[int((1-testing_portion)*len(example_data)):]

        self.training_data = example_data[:int((1-testing_portion)*len(example_data))]

        random.shuffle(self.training_data)
        random.shuffle(self.testing_data)

        self.input_size = self.get_input_size()



    def get_input_size(self):
        return len(self.training_data[0][0][0])

class ReCAConfig(rc_if.ExternalRCConfig):
    """
    Condiguration of the reCA-system

    Various possible configurations:
    1.

    """
    def __init__(self):
        self.reservoir = None
        self.I = 0
        self.classifier = None
        self.encoder = None
        self.time_transition = None
        self.parallelizer = None


    def set_random_mapping_config(self, ca_rule_scheme=None, N=4, R=4, C=3, I=12, classifier="linear-svm", time_transition="random_permutation", mapping_permutations=True):
        #print("Setting config: R:" + str(R) + " C:" +str(C) + " I:" + str(I) + " clf: " + str(classifier) + " rule scheme: " + str(ca_rule_scheme.rule_list))
        ca_size = N*R*C  # Used to create rule scheme
        # sets up elementary CA:
        #print(ca_rule_scheme)
        self.reservoir = ca.ElemCAReservoir(ca_size)
        self.reservoir.set_rule_config(ca_rule_scheme)

        # clf
        if classifier == "linear-svm":
            self.classifier = scikit_clfs.SVM()
        elif classifier == "perceptron_sgd":
            self.classifier = scikit_clfs.SGD()
        elif classifier == "linear_regression":
            self.classifier = scikit_clfs.LinReg()
        elif classifier == "tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder
        self.encoder = enc.RandomMappingEncoder(permutations=mapping_permutations)
        self.encoder.R = R
        self.encoder.C = C

        self.I = I
        if time_transition == "normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()
        elif time_transition == "or":
            self.time_transition = time_trans.ORTimeTransition()
        elif time_transition == "and":
            self.time_transition = time_trans.ANDTimeTransition()
        elif time_transition == "nand":
            self.time_transition = time_trans.NANDTimeTransition()
    def set_uniform_config(self, ca_rule=105, R=4, C=3, I=12, classifier="linear-svm",
                                    encoding="random_mapping", time_transition="random_permutation"):
        # sets up elementary CA:
        self.reservoir = ca.ElemCAReservoir()
        self.reservoir.set_uniform_rule(ca_rule)

        # clf
        if classifier=="linear-svm":
            self.classifier = scikit_clfs.SVM()
        elif classifier =="perceptron_sgd":
            self.classifier = scikit_clfs.SGD()
        elif classifier =="linear_regression":
            self.classifier = scikit_clfs.LinReg()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder
        if encoding == "random_mapping":
            self.encoder = enc.RandomMappingEncoder()
            self.encoder.R = R
            self.encoder.C = C

        self.I = I
        if time_transition=="normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()

    def set_non_uniform_config(self, rule_scheme, R=6, C=4, I=4, classifier="linear-svm", encoding="random_mapping",
                               time_transition="random_permutation"):
        """

        :param rule_scheme: Must be a way to design a non-uniform ca reservoir of the exact same size (R*C*N etc)
        :param R:
        :param C:
        :param I:
        :param classifier:
        :param encoding:
        :param time_transition:
        :return:
        """
        # sets up elementary CA:
        self.reservoir = ca.ElemCAReservoir()
        self.reservoir.set_rules(rule_scheme.get_scheme(4*R*C))
        #self.reservoir.set_uniform_rule(90)


        # clf
        if classifier=="linear-svm":
            self.classifier = scikit_clfs.SVM()
        elif classifier =="perceptron_sgd":
            self.classifier = scikit_clfs.SGD()
        elif classifier =="linear_regression":
            self.classifier = scikit_clfs.LinReg()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder
        if encoding == "random_mapping":
            self.encoder = enc.RandomMappingEncoder()
            self.encoder.R = R
            self.encoder.C = C

        self.I = I
        if time_transition=="normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()

    def set_uniform_margem_config(self, rule_scheme=None,N=4, R_i=4, R=20, I=4, classifier="perceptron_sgd", time_transition="xor"):
        """

        :param rule:
        :param R_i:
        :param R: Padding on each side of the input (buffers)
        :param I:
        :param classifier:
        :param time_transition:
        :return:
        """
        print("Running rotation with values: rule: " + str("unkn. ") + ", R_i: " + str(R_i) + ", R:" + str(R), "I: " +str(I))
        ca_size = N * R_i + R*2  # Used to create rule scheme
        # sets up elementary CA:
        before_time = time.time()
        self.reservoir = ca.ElemCAReservoir(ca_size)
        self.reservoir.set_rule_config(rule_scheme)
        #self.reservoir.set_rules(rule)
        #self.reservoir.set_uniform_rule(rule)




        # clf
        if classifier=="linear-svm":
            self.classifier = scikit_clfs.SVM()
        elif classifier =="perceptron_sgd":
            self.classifier = scikit_clfs.SGD()
        elif classifier == "random_forest":
            self.classifier = scikit_clfs.RandomForest()
        elif classifier =="linear_regression":
            self.classifier = scikit_clfs.LinReg()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder

        self.encoder = enc.RotationEncoder()
        self.encoder.R = R
        self.encoder.R_i = R_i
        self.I = I

        if time_transition == "normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()
        elif time_transition == "or":
            self.time_transition = time_trans.ORTimeTransition()
        elif time_transition == "and":
            self.time_transition = time_trans.ANDTimeTransition()
        elif time_transition == "nand":
            self.time_transition = time_trans.NANDTimeTransition()
class ReCAruleConfig:
    def __init__(self, uniform_rule=None, non_uniform_list=None, non_uniform_individual=None):
        self.uniform = False
        self.non_uniform_list = False
        self.dynamic = False
        self.uniform = False
        self.non_uniform = False
        self.dynamic = False
        if uniform_rule is not None:
            self.uniform = True
            self.uniform_rule = uniform_rule
        elif non_uniform_list is not None:
            self.non_uniform = True
            self.rule_list = non_uniform_list
        elif non_uniform_individual is not None:
            self.dynamic = True
            self.non_uni_ca_ind = non_uniform_individual

    def get_scheme(self, size):
        if self.uniform:
            return [self.uniform_rule for _ in range(size)]
        elif self.non_uniform:
            return self.rule_list
        elif self.dynamic:
            self.non_uni_ca_ind.develop(size)
            return self.non_uni_ca_ind.phenotype.non_uniform_config
        else:
            raise ValueError



















