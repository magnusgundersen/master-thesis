"""
Functionality and experiments.

"""
__author__ = 'magnus'
from reca import reca_system as reCA
from gui import ca_basic_visualizer as bviz
import random
import pprint
import itertools # for permutations
import csv
import os
import pickle as pickle
import time
import experiment_data.data_interpreter as data_int

from multiprocessing import Pool

class Project:
    """
    Contains all tasks and functionality specifically to the specialization project.

    Will communicate with the master, and give the user feedback if neccecery.


    """
    def __init__(self):
        pass

    def img_clf_task(self):
        img_data = self.open_temporal_data("cifar.data")
        reCA_problem = reCA.ReCAProblem(img_data)
        reCA_config = reCA.ReCAConfig()
        reCA_config.set_single_reservoir_config(ca_rule=90, R=4, C=2, I=20, classifier="linear-svm",
                                                encoding="random_mapping",
                                                time_transition="random_permutation")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        #reCA_system.fit_to_problem(9/10)
        reCA_system.tackle_ReCA_problem()
        # reCA_config.encoder.create_mappings(4)

        reCA_out = reCA_system.test_on_problem(9/10)
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0]
        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[i][1]
            _input = "".join([str(x) for x in example_test[i][0]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        # Visualize:
        """
        outputs = reCA_out.all_RCOutputs
        whole_output = []
        lists_of_states = [output.list_of_states for output in outputs]
        for output in lists_of_states:
            width = len(output[0])
            whole_output.extend(output)
            whole_output.extend([[-1 for _ in range(width)]])
        self.visualise_example(whole_output)
        """

        # Visualize:
        outputs = reCA_system.get_example_run()
        whole_output = []
        lists_of_states = [output.list_of_states for output in outputs]
        for output in lists_of_states:
            width = len(output[0])
            new_output = []
            for line in output:
                new_output.append([(-1 if i == 0 else 1) for i in line])

            whole_output.extend(new_output)
            whole_output.extend([[0 for _ in range(width)]])
        self.visualise_example(whole_output)

    def europarl_translation_task(self):

        # Currently only german is implemented
        #translation_data = self.open_temporal_data("en-de.data")
        data_interpreter = self.open_data_interpreter("europarl")

        # reca_prob
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
                                                encoding="random_mapping",
                                                time_transition="random_permutation")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_dynamic_sequence_data()  # tested on the test-set


        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0]
        #print("ex run:"+str(example_run))
        raw_predictions = []
        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            raw_predictions.append(prediction)
            correct = example_test[i][1]
            _input = "".join([str(x) for x in example_test[i][0]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        print("Predicted sentence:" + data_interpreter.convert_from_bit_sequence_to_string(raw_predictions, "german"))

        # Visualize:
        outputs = reCA_system.get_example_run()
        whole_output = []
        lists_of_states = [output.list_of_states for output in outputs]
        for output in lists_of_states:
            width = len(output[0])
            new_output = []
            for line in output:
                new_output.append([(-1 if i == 0 else 1) for i in line])

            whole_output.extend(new_output)
            whole_output.extend([[0 for _ in range(width)]])
        self.visualise_example(whole_output)

    def five_bit_task(self):


        #n_bit_data = self.open_temporal_data("5bit/5_bit_10_dist_32")
        data_interpreter = self.open_data_interpreter("5bit")
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_rule_scheme = reCA.ReCAruleConfig()
        reCA_config = reCA.ReCAConfig()

        reCA_config.set_non_uniform_config(reCA_rule_scheme)
        reCA_system = reCA.ReCASystem()




        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        #reCA_config.encoder.create_mappings(4)

        reCA_out = reCA_system.test_on_problem(0)
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0]
        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[i][1]
            _input = "".join([str(x) for x in example_test[i][0]])
            print("Input: " + _input +"  Correct: " + str(correct) +"  Predicted:" + str(prediction))

        # Visualize:
        outputs = reCA_system.get_example_run()
        whole_output = []
        lists_of_states = [output.list_of_states for output in outputs]
        for output in lists_of_states:
            width = len(output[0])
            new_output = []
            for line in output:
                new_output.append([(-1 if i == 0 else 1) for i in line])

            whole_output.extend(new_output)
            whole_output.extend([[0 for _ in range(width)]])
        self.visualise_example(whole_output)


    def majority_task(self):

        majority_data = self.open_temporal_data("majority/8_bit_mix_1000")


        reCA_problem = reCA.ReCAProblem(majority_data)
        reCA_config = reCA.ReCAConfig()
        reCA_config.set_single_reservoir_config(ca_rule=90, R=16, C=3, I=9, classifier="linear-svm",
                                                encoding="random_mapping", time_transition="random_permutation")

        reCA_config.set_parallel_reservoir_config()

        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()


        reCA_system.fit_to_problem(validation_set_size=0.1)


    def visualise_example(self, training_array):
        visualizer = bviz.CAVisualizer()
        visualizer.visualize(training_array)

    def convert_to_array(self, training_set):
        new_training_set = []
        for _input,_output in training_set:
            new_training_set.append(([int(number) for number in _input],int(_output)))

        return new_training_set

    def open_data(self, filename):
        """
        Reads data from file

        data must be on the form of

        1010010101...100101 0

        Where the first vector is binary, and the last integer is the class. Must also be binary.
        :param filename:
        :return:
        """
        dataset = []
        with open("../experiment_data/"+filename, "r") as f:
            content = f.readlines()
            for line in content:
                _input, _output = line.split(" ")
                dataset.append((_input,_output[0]))
        return dataset

    def open_temporal_data(self, filename):
        dataset = []
        with open("../experiment_data/"+filename, "r") as f:
            content = f.readlines()
            training_set = []
            for line in content:
                if line == "\n":
                    dataset.append(training_set)
                    training_set = []
                else:
                    _input, _output = line.split(" ")
                    training_set.append(([int(number) for number in _input],_output[0:-1]))
        return dataset

    def open_data_interpreter(self, type_of_interpreter):
        if type_of_interpreter == "europarl":
            return data_int.TranslationBuilder()

        if type_of_interpreter == "5bit":
            return data_int.FiveBitBuilder()







