"""
Functionality and experiments.

"""
__author__ = 'magnus'
from reca import reca_system as reCA
from gui import ca_basic_visualizer as bviz
import master.ea_ca as ea_ca
import ea.evoalg as evoalg
import datetime
import ea.adult_selector as adult_select
import ea.parent_selector as parent_select
import ea.individual as ea_ind
import random
import pprint
import itertools # for permutations
import csv
import os
import pickle as pickle
import time
import experiment_data.data_interpreter as data_int
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import json
import gui.graphic_vis as visual

# Global variables
file_location = os.path.dirname(os.path.realpath(__file__))

# Workers:
def open_data_interpreter(type_of_interpreter, **kwargs):
    if type_of_interpreter == "europarl":
        return data_int.TranslationBuilder()

    elif type_of_interpreter == "5bit":
        distractor_period = kwargs.get("distractor_period") if kwargs.get('distractor_period') is not None else 10
        training_ex = kwargs.get("training_ex") if kwargs.get('training_ex') is not None else 32
        testing_ex = kwargs.get("testing_ex") if kwargs.get('testing_ex') is not None else 32
        return data_int.FiveBitBuilder(distractor_period, training_ex=training_ex, testing_ex=testing_ex)

    elif type_of_interpreter == "20bit":
        distractor_period = kwargs.get("distractor_period") if kwargs.get('distractor_period') is not None else 10
        training_ex = kwargs.get("training_ex") if kwargs.get('training_ex') is not None else 120
        testing_ex = kwargs.get("testing_ex") if kwargs.get('testing_ex') is not None else 100
        return data_int.TwentyBitBuilder(distractor_period, training_ex=training_ex, testing_ex=testing_ex)

    elif type_of_interpreter == "japanese_vowels":
        training_ex = kwargs.get("training_ex") if kwargs.get('training_ex') is not None else 270
        testing_ex = kwargs.get("testing_ex") if kwargs.get('testing_ex') is not None else 370
        return data_int.JapaneseVowelsBuilder(training_ex=training_ex, testing_ex=testing_ex)

    elif type_of_interpreter == "5bit_density":
        distractor_period = kwargs.get("distractor_period") if kwargs.get('distractor_period') is not None else 10
        training_ex = kwargs.get("training_ex") if kwargs.get('training_ex') is not None else 32
        testing_ex = kwargs.get("testing_ex") if kwargs.get('testing_ex') is not None else 32
        return data_int.FiveBitAndDensityBuilder(distractor_period, training_ex=training_ex, testing_ex=testing_ex)

    elif type_of_interpreter == "seq_to_seq_synth":
        return data_int.SyntheticSequenceToSequenceBuilder()

    elif type_of_interpreter == "sqrt_seq":
        return data_int.SequenceSquareRootBuilder()

def run_five_bit(data_interpreter, rci_value, classifier, reca_rule, do_mappings):
    #data_interpreter = open_data_interpreter(type_of_interpreter="5bit", distractor_period=200)
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()
    #reCA_rule_scheme = reCA.ReCAruleConfig(uniform_rule=rule)
    reCA_config.set_random_mapping_config(ca_rule_scheme=reca_rule, R=rci_value[0], C=rci_value[1], I=rci_value[2],
                                          classifier=classifier, time_transition="or", mapping_permutations=do_mappings)
    reCA_system = reCA.ReCASystem()
    reCA_system.set_problem(reCA_problem)
    reCA_system.set_config(reCA_config)
    reCA_system.initialize_rc()
    reCA_system.tackle_ReCA_problem()

    reCA_out = reCA_system.test_on_problem()

    #print(""+str(reCA_out.total_correct) + "    " + str(len(reCA_out.all_test_examples)))
    result = int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)
    return result

def random_mapping_worker(data_interpreter, ca_config, reca_rule):
    #data_interpreter = open_data_interpreter(type_of_interpreter="5bit", distractor_period=200)
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()
    #reCA_rule_scheme = reCA.ReCAruleConfig(uniform_rule=rule)
    reCA_config.set_random_mapping_config(ca_rule_scheme=reca_rule,N=reCA_problem.input_size, R=ca_config.get("R"), C=ca_config.get("C"), I=ca_config.get("I"),
                                          classifier=ca_config.get("classifier"), time_transition=ca_config.get("time_transition"),
                                          mapping_permutations=ca_config.get("do_mappings"))
    reCA_system = reCA.ReCASystem()
    reCA_system.set_problem(reCA_problem)
    reCA_system.set_config(reCA_config)
    reCA_system.initialize_rc()
    reCA_system.tackle_ReCA_problem()

    reCA_out = reCA_system.test_on_problem()

    #print(""+str(reCA_out.total_correct) + "    " + str(len(reCA_out.all_test_examples)))
    result = int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)
    return result


# Classes:
class Project:
    """
    Contains all tasks and functionality specifically to the specialization project.

    Will communicate with the master, and give the user feedback if neccecery.


    """
    def __init__(self):
        pass

    #######################
    # Single run problems #
    #######################
    def five_bit_task(self):
        data_interpreter = open_data_interpreter("5bit", distractor_period=10, training_ex=1, testing_ex=1)
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()

        #size = 100*4
        #rule_list = []
        #avail_rules = [110,1, 90,1]
        #no_rules = 4
        #for i in range(no_rules):
        #    rule = random.randint(0,255)
        #    rule = avail_rules[i]
        #    rule_list.extend([rule for _ in range(size//no_rules)])
        #rule_list = [89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 149, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 151, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122]


        with open(file_location+ "/../experiment_data/rules/NuniRule2843_f=1000.ind", "rb") as f:
            evolved_ind = pickle.load(f)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_list=rule_list)
        reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        #reCA_rule = reCA.ReCAruleConfig(uniform_rule=195)
        reCA_config.set_random_mapping_config(reCA_rule, R=80, C=1, I=2, mapping_permutations=False, classifier="perceptron_sgd", time_transition="or")
        #reCA_config.set_non_uniform_config(reCA_rule, R=8, C=5, I=8, classifier="perceptron_sgd")
        #reCA_config.set_uniform_margem_config(rule=[141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18], R_i=2, R=76, I=8, classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_problem()

        # PRINTOUT:
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0]

        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[1][i]
            _input = "".join([str(x) for x in example_test[0][i]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

    def twenty_bit_task(self):
        """
        A run of the 20-bit task
        :return:
        """
        before = time.time()
        data_interpreter = open_data_interpreter("20bit", distractor_period=10, training_ex=120, testing_ex=20)
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        with open(file_location + "/../experiment_data/rules/NuniRule9368_f=1000.ind", "rb") as f:
            evolved_ind = pickle.load(f)
        reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        reCA_rule = reCA.ReCAruleConfig(uniform_rule=90)
        reCA_config = reCA.ReCAConfig()
        #reCA_config.set_non_uniform_config(reCA_rule_scheme)
        reCA_config.set_random_mapping_config(reCA_rule, N=reCA_problem.input_size, R=12, C=40, I=2, classifier="perceptron_sgd", time_transition="random_permutation")
        #reCA_config.set_uniform_margem_config(reCA_rule, N=reCA_problem.input_size, R=(20*(10+10+10)), R_i=2, I=20, classifier="perceptron_sgd", time_transition="xor")
        reCA_system = reCA.ReCASystem()


        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()


        reCA_out = reCA_system.test_on_problem()
        print("time usage: " + str(time.time()- before))
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0][1]

        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[i]
            _input = "".join([str(x) for x in example_test[i]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

    def japanese_vowels(self):

        # Currently only german is implemented
        #translation_data = self.open_temporal_data("en-de.data")
        data_interpreter = open_data_interpreter("japanese_vowels")

        # reca_prob
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        #reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
        #                                        encoding="random_mapping",
        #                                        time_transition="random_permutation")
        with open(file_location+ "/../experiment_data/rules/NuniRule6422_f=910.ind", "rb") as f:
            evolved_ind = pickle.load(f)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_list=rule_list)
        reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        reCA_rule = reCA.ReCAruleConfig(uniform_rule=90)
        #reCA_config.set_uniform_margem_config(rule_scheme=reCA_rule, N=reCA_problem.input_size, R=(reCA_problem.input_size*2)+29*4, R_i=2, I=4)
        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=reCA_problem.input_size, R=40, C=1, I=20, classifier="perceptron_sgd",
                                              mapping_permutations=True, time_transition="or")
        #reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=14*2, R=64, C=1, I=4, time_transition="xor", classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        #return

        reCA_out = reCA_system.test_semi_dynamic_sequence_data()  # tested on the test-set


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
            correct = example_test[1][i]
            _input = "".join([str(x) for x in example_test[0][i]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        #print("Predicted sentence:" + data_interpreter.convert_from_bit_sequence_to_string(raw_predictions, "german"))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

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
        data_interpreter = open_data_interpreter("europarl")

        # reca_prob
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        # reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
        #                                        encoding="random_mapping",
        #                                        time_transition="random_permutation")
        with open(file_location + "/../experiment_data/rules/NuniRule6061_f=968.ind", "rb") as f:
            evolved_ind = pickle.load(f)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_list=rule_list)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        reCA_rule = reCA.ReCAruleConfig(uniform_rule=90)

        # English alphabet size: 54
        #reCA_config.set_uniform_margem_config(rule_scheme=reCA_rule, N=54, R=10, R_i=1, I=2)
        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=54, R=4, C=4, I=6,
                                              classifier="perceptron_sgd")
        # reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=14*2, R=64, C=1, I=4, time_transition="xor", classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_fully_dynamic_sequence_data()  # tested on the test-set


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
            try:
                correct = example_test[1][i]
            except:
                correct = "-"*57

            try:
                _input = "".join([str(x) for x in example_test[0][i]])
            except:
                _input = "00000000000000000000000000000000000000000000000000001"

            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        print("Predicted sentence:" + data_interpreter.convert_from_bit_sequence_to_string(raw_predictions, "german"))
        pointer = 0
        print()
        print("All test-sentences: ")
        for predictions in reCA_out.all_predictions:
            raw_prediction = [pred[0] for pred in predictions]
            sentence = data_interpreter.convert_from_bit_sequence_to_string(raw_prediction, "german")

            correct_sentence = [corr for corr in reCA_out.all_test_examples[pointer][1]]
            correct_sentence = data_interpreter.convert_from_bit_sequence_to_string(correct_sentence, "german")

            from_sentence = ["".join([str(char) for char in from_sentence]) for from_sentence in reCA_out.all_test_examples[pointer][0]]
            from_sentence = data_interpreter.convert_from_bit_sequence_to_string(from_sentence, "english")

            print("From sentence: " + str(from_sentence))
            print("To sentence  : " + str(correct_sentence))
            print("Predicted    : " + str(sentence))
            print("---")
            pointer += 1


        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

    def five_bit_density_task(self):
        data_interpreter = open_data_interpreter("5bit_density", distractor_period=10, training_ex=400, testing_ex=100)
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()

        with open(file_location + "/../experiment_data/rules/NuniRule2007_f=1000.ind", "rb") as f:
            evolved_ind = pickle.load(f)

        reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)

        reCA_rule = reCA.ReCAruleConfig(uniform_rule=90)

        reCA_config.set_random_mapping_config(reCA_rule, N=reCA_problem.input_size, R=28, C=6, I=4, classifier="perceptron_sgd",
                                              time_transition="random_permutation", mapping_permutations=True
                                                              )
        # reCA_config.set_non_uniform_config(reCA_rule, R=8, C=5, I=8, classifier="perceptron_sgd")
        # reCA_config.set_uniform_margem_config(rule=[141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18], R_i=2, R=76, I=8, classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_problem()

        # PRINTOUT:
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))
        print("--example--")
        example_run = reCA_out.all_predictions[0]
        example_test = reCA_out.all_test_examples[0]

        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[1][i]
            _input = "".join([str(x) for x in example_test[0][i]])
            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

    def sequence_to_sequence_synth_task(self):
        # Currently only german is implemented
        #translation_data = self.open_temporal_data("en-de.data")
        data_interpreter = open_data_interpreter("seq_to_seq_synth")

        # reca_prob
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        # reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
        #                                        encoding="random_mapping",
        #                                        time_transition="random_permutation")
        #with open(file_location + "/../experiment_data/rules/NuniRule6061_f=968.ind", "rb") as f:
        #    evolved_ind = pickle.load(f)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_list=rule_list)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        reCA_rule = reCA.ReCAruleConfig(uniform_rule=150)

        #
        #reCA_config.set_uniform_margem_config(rule_scheme=reCA_rule, N=54, R=10, R_i=1, I=2)
        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=reCA_problem.input_size, R=12, C=6, I=4,
                                              classifier="perceptron_sgd")
        # reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=14*2, R=64, C=1, I=4, time_transition="xor", classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_fully_dynamic_sequence_data()  # tested on the test-set

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
            try:
                correct = example_test[1][i]
            except:
                correct = "-"*len(reCA_problem.prediction_end_signal)

            try:
                _input = "".join([str(x) for x in example_test[0][i]])
            except:
                _input = "".join([str(x) for x in reCA_problem.prediction_input_signal])

            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        #print("Predicted sentence:" + data_interpreter.convert_from_bit_sequence_to_string(raw_predictions, "german"))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)

    def square_root_sequence_task(self):
        # Currently only german is implemented
        #translation_data = self.open_temporal_data("en-de.data")
        data_interpreter = open_data_interpreter("sqrt_seq")

        # reca_prob
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        # reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
        #                                        encoding="random_mapping",
        #                                        time_transition="random_permutation")
        #with open(file_location + "/../experiment_data/rules/NuniRule6061_f=968.ind", "rb") as f:
        #    evolved_ind = pickle.load(f)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_list=rule_list)
        #reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=evolved_ind)
        reCA_rule = reCA.ReCAruleConfig(uniform_rule=90)

        #
        #reCA_config.set_uniform_margem_config(rule_scheme=reCA_rule, N=54, R=10, R_i=1, I=2)
        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=reCA_problem.input_size, R=20, C=1, I=2,
                                              classifier="perceptron_sgd", time_transition="or")
        # reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule, N=14*2, R=64, C=1, I=4, time_transition="xor", classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_fully_dynamic_sequence_data()  # tested on the test-set

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
            try:
                correct = example_test[1][i]
            except:
                correct = "-"*len(reCA_problem.prediction_end_signal)

            try:
                _input = "".join([str(x) for x in example_test[0][i]])
            except:
                _input = "".join([str(x) for x in reCA_problem.prediction_input_signal])

            print("Input: " + _input + "  Correct: " + str(correct) + "  Predicted:" + str(prediction))

        #print("Predicted sentence:" + data_interpreter.convert_from_bit_sequence_to_string(raw_predictions, "german"))

        # Visualize:
        outputs = reCA_system.get_example_run()
        visual.visualize_example_run(outputs)


    ######################
    # Evolved Non-uni CA #
    ######################
    def evolve_ca_five_bit(self):
        # ReCA params
        C = 1
        Rs = [80]
        I = 2
        N = 4
        time_transition = "or"
        classifier = "perceptron_sgd"
        do_mappings = False
        number_of_rules_list = [10]  # Maximum number of distinct rules

        # EA params
        pop_size = 7*2  # Adapt to number of cores
        max_no_generations = 10000
        tests_per_individual = 4
        fitness_threshold_value = 1000
        retest_threshold = 999
        retests_per_individual = 10
        continue_from_checkpoint = True

        print_est = False
        before = time.time()
        for number_of_rules in number_of_rules_list:
            for R in Rs:
                reca_config = {
                    "N": N,
                    "R": R,
                    "I": I,
                    "C": C,
                    "do_mappings": do_mappings,
                    "time_transition": time_transition,
                    "classifier": classifier,

                }

                ea_config = {
                    "number_of_rules": number_of_rules,
                    "pop_size": pop_size,
                    "max_gens": max_no_generations,
                    "fitness_threshold": fitness_threshold_value,
                    "tests_per_individual": tests_per_individual,
                    "retest_threshold": retest_threshold,
                    "retests_per_individual": retests_per_individual,
                }
                ea_problem = ea_ca.NonUni5BitProblem(reca_config, ea_config)

                self.evolve_non_uniform_ca(reca_config, ea_problem, continue_from_checkpoint)
            if print_est:
                ts = time.time()
                print("Time now : " + str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
                print("Time used: " + str(ts - before))
                print("Total time est.: " + str((ts - before) * (len(number_of_rules_list))))
                print("Time est. left : " + str((ts - before) * (len(number_of_rules_list) - 1)))
                print_est = False
            #test_all_rules()

        print("Actual time usage (ea evolve): " + str(time.time() - before))

    def evolve_ca_twenty_bit(self):
        # ReCA params
        C = 16
        R = 12
        I = 2
        N = 7
        time_transition = "random_permutation"
        classifier = "perceptron_sgd"
        do_mappings = True  # If the mappings should be permuted
        number_of_rules = 12  # Maximum number of distinct rules

        # EA params
        pop_size = 7 * 2  # Adapt to number of cores
        max_no_generations = 10000
        tests_per_individual = 2
        fitness_threshold_value = 1000
        retest_threshold = 999
        retests_per_individual = 6

        continue_from_checkpoint = False

        reca_config = {
            "N": N,
            "R": R,
            "I": I,
            "C": C,
            "do_mappings": do_mappings,
            "time_transition": time_transition,
            "classifier": classifier,

        }

        ea_config = {
            "number_of_rules": number_of_rules,
            "pop_size": pop_size,
            "max_gens": max_no_generations,
            "fitness_threshold": fitness_threshold_value,
            "tests_per_individual": tests_per_individual,
            "retest_threshold": retest_threshold,
            "retests_per_individual": retests_per_individual,
        }
        ea_problem = ea_ca.NonUni20BitProblem(reca_config, ea_config)
        self.evolve_non_uniform_ca(reca_config, ea_problem, continue_from_checkpoint)

    def evolve_ca_jap_vowels(self):
        # ReCA params
        C = 16
        R = 12
        I = 2
        N = 14*4
        time_transition = "random_permutation"
        classifier = "perceptron_sgd"
        do_mappings = True  # If the mapping should be permuted
        number_of_rules = 6  # Maximum number of distinct rules

        # EA params
        pop_size = 7 * 2  # Adapt to number of cores
        max_no_generations = 10000
        tests_per_individual = 1
        fitness_threshold_value = 1000
        retest_threshold = 999
        retests_per_individual = 1

        continue_from_checkpoint = False

        reca_config = {
            "N": N,
            "R": R,
            "I": I,
            "C": C,
            "do_mappings": do_mappings,
            "time_transition": time_transition,
            "classifier": classifier,

        }

        ea_config = {
            "number_of_rules": number_of_rules,
            "pop_size": pop_size,
            "max_gens": max_no_generations,
            "fitness_threshold": fitness_threshold_value,
            "tests_per_individual": tests_per_individual,
            "retest_threshold": retest_threshold,
            "retests_per_individual": retests_per_individual,
                }
        ea_problem = ea_ca.NonUniCAJapVowsProblem(reca_config, ea_config)
        self.evolve_non_uniform_ca(reca_config, ea_problem, continue_from_checkpoint)

    def evolve_ca_five_bit_and_density(self):
        # ReCA params
        C = 1
        R = 800
        I = 2
        N = 4+1 # Five bit signals + majority signals
        time_transition = "or"
        classifier = "perceptron_sgd"
        do_mappings = False  # If the mappings should be permuted
        number_of_rules = 10  # Maximum number of distinct rules

        # EA params
        pop_size = 7 * 1   # Adapt to number of cores
        max_no_generations = 10000
        tests_per_individual = 2
        fitness_threshold_value = 1000
        retest_threshold = 999
        retests_per_individual = 10

        continue_from_checkpoint = False

        reca_config = {
            "N": N,
            "R": R,
            "I": I,
            "C": C,
            "do_mappings": do_mappings,
            "time_transition": time_transition,
            "classifier": classifier,

        }

        ea_config = {
            "number_of_rules": number_of_rules,
            "pop_size": pop_size,
            "max_gens": max_no_generations,
            "fitness_threshold": fitness_threshold_value,
            "tests_per_individual": tests_per_individual,
            "retest_threshold": retest_threshold,
            "retests_per_individual": retests_per_individual,
        }
        ea_problem = ea_ca.NonUni5BitandDensityProblem(reca_config, ea_config)
        self.evolve_non_uniform_ca(reca_config, ea_problem, continue_from_checkpoint)

    def evolve_non_uniform_ca(self, ca_config, ea_prob, continue_from_ckp=False):

        #nonUniCAprob = ea_ca.NonUniCAProblem(ca_config, fitness_threshold=fitness_threshold_value, init_pop_size=pop_size,
        #                                     max_number_of_generations=max_generations,
        #                                     allowed_number_of_rules=allowed_distinct_rules,
        #                                     test_per_ind=tests_per_ind)

        ea = evoalg.EA()

        ea_output = ea.solve(ea_prob, saved_state=continue_from_ckp)

        # pickle.dump(ea_output, open("ea.pkl", "wb"))
        run_name = "earun_R" + str(ca_config.get("R")) + "C" + str(ca_config.get("C")) + "I" + str(ca_config.get("I")) + \
                   "_rules" + str(ea_prob.allowed_number_of_rules) + "_popsize" + str(ea_prob.init_pop_size) + \
                   "_gens" + str(ea_prob.max_number_of_generations)

        pickle.dump(ea_output, open(file_location + "/../experiment_data/ea_runs/" + run_name + ".ea_output", "wb"))
        best_individual = ea_output.best_individual
        best_individ_scheme = best_individual.phenotype.non_uniform_config
        print(best_individual.serialize())
        non_uni_rule_serialize = {}
        non_uni_rule_serialize["full_size_rule_list"] = best_individ_scheme
        non_uni_rule_serialize["raw rule"] = best_individual.genotype.rule_scheme
        ea_data = {"ca_config": ca_config,
                   "popsize": ea_prob.init_pop_size,
                   "max_gens": ea_prob.max_number_of_generations,
                   "test per ind": ea_prob.test_per_ind,
                   "allowed number of rules": ea_prob.allowed_number_of_rules,
                   }
        non_uni_rule_serialize["ea_data"] = ea_data

        with open(file_location + "/../experiment_data/rules/" + run_name + "JSON.json", "w") as outfile:
            json.dump(non_uni_rule_serialize, outfile, sort_keys = True, indent = 4)
        with open(file_location + "/../experiment_data/rules/" + str(best_individual) + ".ind", "wb") as outfile:
            pickle.dump(best_individual, outfile)
        visual.make_fitnessgraph(ea_output, run_name)

    ########################
    # Testing and batching #
    ########################

    def test_all_rules(self):
        uni_rules = []
        #uni_rules = [89, 149, 151, 57, 133, 196, 101, 120, 20, 122]
        uni_rules = [90, 150, 22]
        #uni_rules = [i for i in range(256)]


        with open(file_location + "/../experiment_data/rules/NuniRule2843_f=1000.ind", "rb") as f:
            evolved_ind = pickle.load(f)
        #nuni_rules = [evolved_ind]
        nuni_rules = []
        """
        json_rule_files = []
        file_location = os.path.dirname(os.path.realpath(__file__))
        all_files = os.listdir(file_location+"/../experiment_data/ea_runs")
        for file in all_files:
            if str(file).lower().endswith(".json"):
                json_rule_files.append(file)

        json_data = []
        for file in json_rule_files:
            with open(file_location +"/../experiment_data/ea_runs/"+file, "r") as outfile:
                json_data.append(json.load(outfile))
        nuni_rules = {}

        for data in json_data:
            #print(data)
            ea_data = data.get('ea_data')
            number_of_distinct_rules = ea_data.get('allowed number of rules')
            if nuni_rules.get("nuni=" + str(number_of_distinct_rules)) is None:
                nuni_rules["nuni=" + str(number_of_distinct_rules)] = {}
            nuni_rules["nuni=" + str(number_of_distinct_rules)][ea_data.get("R")] = data.get("full_size_rule_list")
        """
        project = Project()
        project.test_rules(uni_rules, nuni_rules)

    def test_rules(self, uni_rules, non_uni_rules, task="20 bit"):
        Rs = [12]
        C = 16
        I = 2

        RCI_values_r_change = [(x, C, I) for x in Rs]
        print("testing rules")
        RCI_values = RCI_values_r_change
        # RCI_values = [(1,1,1)]
        # distractor_periods = [10, 50, 100, 200]
        # distractor_periods = [10, 25, 50]
        threads = 7
        number_of_tests = threads * 2


        plotconfigs = {}
        for rule in uni_rules:
            mappings = True
            print("Testing rule: "+ str(rule))
            r_dict = {}
            r_list = []
            for rci_value in RCI_values:
                if task =="5 bit":
                    reCA_rule = reCA.ReCAruleConfig(uniform_rule=rule)
                    data_interpreter = open_data_interpreter("5bit", distractor_period=10)
                    with multiprocessing.Pool(threads) as p:
                        results = p.starmap(run_five_bit,
                                            [(data_interpreter, rci_value, "perceptron_sgd", reCA_rule, mappings) for _ in
                                             range(number_of_tests)])

                    r_dict["R=" + str(rci_value[0])] = results
                    r_list.append(int(np.mean(results)))

                elif task== "20 bit":
                    reca_config = {
                        "N": 7,
                        "R": rci_value[0],
                        "I": rci_value[2],
                        "C": rci_value[1],
                        "do_mappings": True,
                        "time_transition": "random_permutation",
                        "classifier": "perceptron_sgd",
                    }

                    reCA_rule = reCA.ReCAruleConfig(uniform_rule=rule)
                    data_interpreter = open_data_interpreter("20bit", distractor_period=10)
                    with multiprocessing.Pool(threads) as p:
                        results = p.starmap(random_mapping_worker,
                                            [(data_interpreter, reca_config, reCA_rule) for _
                                             in
                                             range(number_of_tests)])

                    r_dict["R=" + str(rci_value[0])] = results
                    r_list.append(int(np.mean(results)))
            with open(file_location + "/../experiment_data/rule_testing/rule_" + str(rule) + "_allinfo_JSON.json",
                      "w") as outfile:
                json.dump(r_dict, outfile)

            plotconfigs["Rule " + str(rule)] = r_list
        for rule in non_uni_rules:
            mappings= False
            r_dict = {}
            r_list = []
            for rci_value in RCI_values:
                if task=="5 bit":
                    data_interpreter = open_data_interpreter("5bit", distractor_period=1000)
                    reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=rule)
                    with multiprocessing.Pool(threads) as p:
                        results = p.starmap(run_five_bit,
                                            [(data_interpreter, rci_value, "perceptron_sgd", reCA_rule, mappings) for _ in
                                             range(number_of_tests)])

                    r_dict["R=" + str(rci_value[0])] = results
                    r_list.append(int(np.mean(results)))

                elif task == "20 bit":
                    reca_config = {
                        "N": 7,
                        "R": rci_value[0],
                        "I": rci_value[2],
                        "C": rci_value[1],
                        "do_mappings": True,
                        "time_transition": "random_permutation",
                        "classifier": "perceptron_sgd",
                    }

                    reCA_rule = reCA.ReCAruleConfig(non_uniform_individual=rule)
                    data_interpreter = open_data_interpreter("20bit", distractor_period=10)
                    with multiprocessing.Pool(threads) as p:
                        results = p.starmap(random_mapping_worker,
                                            [(data_interpreter, reca_config, reCA_rule) for _
                                             in
                                             range(number_of_tests)])

                    r_dict["R=" + str(rci_value[0])] = results
                    r_list.append(int(np.mean(results)))
            with open(file_location + "/../experiment_data/rule_testing/rule_" + str(rule) + "_allinfo_JSON.json",
                      "w") as outfile:
                json.dump(r_dict, outfile)
            plotconfigs["Rule " + str(rule)] = r_list

        #print(plotconfigs)
        for w in sorted(plotconfigs, key=plotconfigs.get, reverse=True):
            print(w, plotconfigs[w])
        with open(file_location + "/../experiment_data/rule_testing/full_plotconfig.json", "w") as outfile:
            json.dump(plotconfigs, outfile)

        visual.create_graph_from_jsonconfig(file_location + "/../experiment_data/rule_testing/full_plotconfig.json", Rs)








