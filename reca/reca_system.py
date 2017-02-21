"""
ReCA-system that is implemented.

"""
__author__ = 'magnus'
from classifier import skl_svm as svm
from classifier import tfl_ann as tflann
from reservoir import ca as ca
from reservoircomputing import rc as rc
from reservoircomputing import rc_interface as rc_if
from reca import encoder as enc
from reca import time_transition_function as time_trans
import random


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

    def tackle_ReCA_problem(self, validation_set_size=0.5):
        # 1. Sequential or not??
        # 2. Seq to seq? Or sequential classification?
        # All problems at this stage must be binary!
        if self.reCA_problem is None:
            raise ValueError("No ReCAProblem set!")

        self.example_data = None
        # Feed-Forward task (Like plain CIFAR):

        training_data = self.reCA_problem.training_data

        # For each input.
        for training_example in training_data:
            output = self.rc_framework.fit_to_data(training_example)  # Only one timestep
            if self.example_data is None:
                self.example_data = output

        self.rc_framework.train_classifier()



    def fit_to_problem(self, validation_set_size=0.5):
        """

        :return:
        """
        if self.reCA_problem is None:
            raise ValueError("No reCAProblem set!")

        training_data = self.reCA_problem.training_data[:int(len(self.reCA_problem.training_data)*validation_set_size)]


        self.example_data = None
        # Run each training-example through the rc-framework
        for training_example in training_data:
            #  We now have a timeseries of data, on which the rc-framework must be fitted
            output = self.rc_framework.fit_to_data(training_example)
            if self.example_data is None:
                self.example_data = output

        self.rc_framework.train_classifier()
        return 0, 0


    def test_on_dynamic_sequence_data(self):
        """
                :return:

                """
        if self.reCA_problem is None:
            raise ValueError("No reCAProblem set!")

        reCA_output = ReCAOutput()
        reCA_output.reCA_config = self.reCA_config

        # divide training_data:
        test_data = self.reCA_problem.testing_data
        pred_stop_signal = self.reCA_problem.pred_end_signal
        reCA_output.all_test_examples = test_data

        number_of_correct = 0

        for test_ex in test_data:
            #  We now have a timeseries of data, on which the rc-framework must be tested
            outputs = self.rc_framework.predict_dynamic_sequence(test_ex, pred_stop_signal)
            reCA_output.all_RCOutputs.append(outputs)
            print(reCA_output.all_predictions)
            print("RBEAKING")
            print("Size of predictions: " + str(len(outputs)))
            print("size of correctssss: " + str(len(test_ex)))


            pointer = 0
            all_correct = True
            predictions = []
            for _, output in test_ex:
                predictions.append(outputs[pointer])
                if output != outputs[pointer]:
                    # print("WRONG: " + str(output) + str( "  ") + str(outputs[pointer]))
                    all_correct = False
                pointer += 1
            reCA_output.all_predictions.append(predictions)

            if all_correct:
                number_of_correct += 1

        # print("Number of correct: " + str(number_of_correct) +" of " + str(len(test_data)))

        reCA_output.total_correct = number_of_correct
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
        self.training_data = []
        self.testing_data = []


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

        first_example_sequence_length = len(example_data[0])
        for data in example_data:
            if len(data) == first_example_sequence_length:
                self.is_fixed_sequence = True

            else:
                self.is_dynamic_sequence = True
                self.is_fixed_sequence = False
                self.pred_end_signal = data_interpreter.get_pred_end_signal()
                break  # we know at least some training-examples are of different lengths

        self.testing_data = data_interpreter.get_testing_data()

        if len(self.testing_data) == 0:
            testing_portion = 0.1
            self.testing_data = example_data[int((1-testing_portion)*len(example_data)):]

        self.training_data = example_data[:int((1-testing_portion)*len(example_data))]

        random.shuffle(self.training_data)
        random.shuffle(self.testing_data)





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

    def set_single_reservoir_config(self, ca_rule=105, R=4, C=3, I=12, classifier="linear-svm",
                                    encoding="random_mapping", time_transition="normalized_addition"):
        # sets up elementary CA:
        self.reservoir = ca.ElemCAReservoir()
        ca_rule = [ca_rule]  # Parallel?
        self.reservoir.set_rules(ca_rule)

        self.parallelizer = enc.ParallelNonUniformEncoder(self.reservoir.rules, "unbounded")

        # clf
        if classifier=="linear-svm":
            self.classifier = svm.SVM()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()


        # Encoder
        if encoding == "random_mapping":
            self.encoder = enc.RandomMappingEncoder(self.parallelizer)
            self.encoder.R = R
            self.encoder.C = C
        elif encoding == "r_is":
            # TODO: R_i-style mapping from Margem (2016)
            pass

        # Padding
        if True:
            # TODO: Padding at each side of the "input".
            # the values at padding is directly copied from time-step to time-step.
            pass


        self.I = I
        if time_transition=="normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()

    def set_parallel_reservoir_config(self, ca_rules=(105,110), parallel_size_policy="unbounded", R=4, C=3, I=12,
                                      classifier="linear-svm", encoding="random_mapping",
                                      time_transition="normalized_addition"):

        # sets up elementary CA:
        self.reservoir = ca.ElemCAReservoir()
        self.reservoir.set_rules(ca_rules)

        #if parallel_size_policy
        self.parallelizer = enc.ParallelNonUniformEncoder(self.reservoir.rules, parallel_size_policy)


        # clf
        if classifier=="linear-svm":
            self.classifier = svm.SVM()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder
        if encoding == "random_mapping":
            self.encoder = enc.RandomMappingEncoder(self.parallelizer)
            self.encoder.R = R
            self.encoder.C = C

        self.I = I
        if time_transition=="normalized_addition":
            self.time_transition = enc.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = enc.RandomPermutationTransition()

    def set_non_uniform_config(self, rule_scheme, R=2, C=3, I=2, classifier="linear-svm", encoding="random_mapping",
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
        self.reservoir.set_rules(rule_scheme.get_scheme())
        #self.reservoir.set_uniform_rule(90)



        # clf
        if classifier=="linear-svm":
            self.classifier = svm.SVM()
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

    def set_uniform_margem_config(self, rule=90, R_i=2, R=240, I=6, classifier="linear-svm", time_transition="xor"):
        """

        :param rule:
        :param R_i:
        :param R: Padding on each side of the input (buffers)
        :param I:
        :param classifier:
        :param time_transition:
        :return:
        """
        print("Running rotation with values: rule: " + str(rule) + ", R_i: " + str(R_i) + ", R:" + str(R), "I: " +str(I))
        # sets up elementary CA:
        self.reservoir = ca.ElemCAReservoir()
        self.reservoir.set_uniform_rule(rule)




        # clf
        if classifier=="linear-svm":
            self.classifier = svm.SVM()
        elif classifier =="tlf_ann":
            self.classifier = tflann.ANN()

        # Encoder

        self.encoder = enc.RotationEncoder()
        self.encoder.R = R
        self.encoder.R_i = R_i
        self.I = I

        if time_transition=="normalized_addition":
            self.time_transition = time_trans.RandomAdditionTimeTransition()
        elif time_transition == "random_permutation":
            self.time_transition = time_trans.RandomPermutationTransition()
        elif time_transition == "xor":
            self.time_transition = time_trans.XORTimeTransition()
class ReCAruleConfig:
    def __init__(self, ca_size):
        self.ca_size = ca_size


    def get_scheme(self):
        rule_list = []
        #keys = [(i, i+1) for i in range(self.ca_size)]   OLD STYLE

        for _ in range(self.ca_size):
            rule_list.append(random.choice([90,150,110,60]))
        return rule_list  # Dummy return



















