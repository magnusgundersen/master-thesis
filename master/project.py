"""
Functionality and experiments.

"""
__author__ = 'magnus'
from reca import reca_system as rcca
from gui import ca_basic_visualizer as bviz
import random
import pprint
import itertools # for permutations
import csv
import os
import pickle as pickle
import time

from multiprocessing import Pool

class Project:
    """
    Contains all tasks and functionality specifically to the specialization project.

    Will communicate with the master, and give the user feedback if neccecery.


    """
    def __init__(self):
        pass


    def n_bit_task(self, n=5):


        n_bit_data = self.open_temporal_data("temp_n_bit/5_bit_15_dist_32")
        rcca_problem = rcca.ReCAProblem(n_bit_data)
        rcca_config = rcca.ReCAConfig()
        rcca_config.set_single_reservoir_config(ca_rule=110, R=8, C=5, I=4, classifier="linear-svm",
                                                        encoding="random_mapping",
                                                        time_transition="random_permutation")
        #rcca_config.set_parallel_reservoir_config(ca_rules=[90,105], parallel_size_policy="bounded", R=8, C=5, I=4,
        #                              classifier="linear-svm", encoding="random_mapping",
        #                              time_transition="random_permutation")

        rcca_system = rcca.ReCASystem()




        rcca_system.set_problem(rcca_problem)
        rcca_system.set_config(rcca_config)
        rcca_system.initialize_rc()
        rcca_system.fit_to_problem(1)

        #rcca_config.encoder.create_mappings(4)

        rcca_out = rcca_system.test_on_problem(0)
        print(str(rcca_out.total_correct) + " of " + str(len(rcca_out.all_test_examples)))
        print("--example--")
        example_run = rcca_out.all_predictions[0]
        example_test = rcca_out.all_test_examples[0]
        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[i][1]
            _input = "".join([str(x) for x in example_test[i][0]])
            print("Input: " + _input +"  Correct: " + str(correct) +"  Predicted:" + str(prediction))

        # Visualize:
        """
        outputs = rcca_out.all_RCOutputs
        whole_output = []
        lists_of_states = [output.list_of_states for output in outputs]
        for output in lists_of_states:
            width = len(output[0])
            whole_output.extend(output)
            whole_output.extend([[-1 for _ in range(width)]])
        self.visualise_example(whole_output)
        """

        # Visualize:
        outputs = rcca_system.get_example_run()
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


        rcca_problem = rcca.RCCAProblem(majority_data)
        rcca_config = rcca.RCCAConfig()
        rcca_config.set_single_reservoir_config(ca_rule=90, R=16, C=3, I=9, classifier="linear-svm",
                                                encoding="random_mapping", time_transition="random_permutation")

        rcca_config.set_parallel_reservoir_config()

        rcca_system = rcca.RCCASystem()

        rcca_system.set_problem(rcca_problem)
        rcca_system.set_config(rcca_config)
        rcca_system.initialize_rc()

        rcca_system.fit_to_problem(validation_set_size=0.1)


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






