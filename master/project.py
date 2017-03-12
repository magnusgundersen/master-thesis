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
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import json

def run_five_bit(data_interpreter, rci_value, classifier, rule=90):
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()

    reCA_config.set_uniform_config(ca_rule=rule, R=rci_value[0], C=rci_value[1], I=rci_value[2], classifier=classifier)
    reCA_system = reCA.ReCASystem()
    reCA_system.set_problem(reCA_problem)
    reCA_system.set_config(reCA_config)
    reCA_system.initialize_rc()
    reCA_system.tackle_ReCA_problem()

    reCA_out = reCA_system.test_on_problem()


    result =  int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)
    return result

def run_non_uniform_five_bit(data_interpreter, rci_value, classifier, rule_scheme):
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()
    rule_config = reCA.ReCAruleConfig(rule_scheme)

    reCA_config.set_non_uniform_config(rule_scheme=rule_config, R=rci_value[0], C=rci_value[1], I=rci_value[2], classifier=classifier)
    reCA_system = reCA.ReCASystem()
    reCA_system.set_problem(reCA_problem)
    reCA_system.set_config(reCA_config)
    reCA_system.initialize_rc()
    reCA_system.tackle_ReCA_problem()

    reCA_out = reCA_system.test_on_problem()


    result =  int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)
    return result

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
        #reCA_config.set_single_reservoir_config(ca_rule=90, R=2, C=3, I=16, classifier="linear-svm",
        #                                        encoding="random_mapping",
        #                                        time_transition="random_permutation")
        reCA_config.set_uniform_margem_config()
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
            correct = example_test[1][i]
            _input = "".join([str(x) for x in example_test[0][i]])
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

        data_interpreter = self.open_data_interpreter("5bit")
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()
        reCA_rule = reCA.ReCAruleConfig([141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18])
        reCA_config.set_uniform_config(ca_rule=90, R=16, C=5, I=4, classifier="perceptron_sgd")
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
        example_test = reCA_out.all_test_examples[0][1]

        for i in range(len(example_run)):
            time_step = example_run[i]
            prediction = time_step[0]
            correct = example_test[i]
            _input = "".join([str(x) for x in example_test[i]])
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

    def twenty_bit_task(self):
        data_interpreter = self.open_data_interpreter("20bit")
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_rule_scheme = reCA.ReCAruleConfig(1)
        reCA_config = reCA.ReCAConfig()

        #reCA_config.set_non_uniform_config(reCA_rule_scheme)
        reCA_config.set_uniform_margem_config()
        reCA_system = reCA.ReCASystem()




        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()


        reCA_out = reCA_system.test_on_problem()
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

    def visualise_example(self, training_array):
        visualizer = bviz.CAVisualizer()
        visualizer.visualize(training_array)

    def open_data_interpreter(self, type_of_interpreter, distractor_period=10):
        if type_of_interpreter == "europarl":
            return data_int.TranslationBuilder()

        elif type_of_interpreter == "5bit":
            return data_int.FiveBitBuilder(distractor_period)

        elif type_of_interpreter == "20bit":
            return data_int.TwentyBitBuilder()

    def test_rules(self, uni_rules, non_uni_rules):
        Rs= [16]
        C = 5
        I = 4
        RCI_values_r_change = [(x, C, I) for x in Rs]
        #uni_rules = [90, 105, 150, 165]
        #non_uni_rules = {"nuni=2":
        #                {2: [random.choice([90,110]) for _ in range(2*C*4)],
        #                 4: [random.choice([90,110]) for _ in range(4*C*4)],
        #                 6: [random.choice([90,110]) for _ in range(6*C*4)],
        #                 8: [random.choice([90,110]) for _ in range(8*C*4)]},
        #                 "nuni=3":
        #                 {2: [random.choice([90,110, 150]) for _ in range(2 * C * 4)],
        #                  4: [random.choice([90,110, 150]) for _ in range(4 * C * 4)],
        #                  6: [random.choice([90,110, 150]) for _ in range(6 * C * 4)],
        #                  8: [random.choice([90,110, 150]) for _ in range(8 * C * 4)]},
        #                 }

        RCI_values = RCI_values_r_change
        #RCI_values = [(1,1,1)]
        #distractor_periods = [10, 50, 100, 200]
        #distractor_periods = [10, 25, 50]
        threads = 8
        number_of_tests = threads*20

        file_location = os.path.dirname(os.path.realpath(__file__))

        plotconfigs = {}
        plotlabels = []
        for rule in uni_rules:
            r_dict = {}
            r_list = []
            for rci_value in RCI_values:

                data_interpreter = self.open_data_interpreter("5bit", distractor_period=10)
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(run_five_bit,
                                        [(data_interpreter, rci_value, "perceptron_sgd", rule) for _ in range(number_of_tests)])

                r_dict["R=" + str(rci_value[0])] = results
                r_list.append(int(np.mean(results)))
            with open(file_location +"/../experiment_data/rule_testing/rule_" + str(rule) +"_allinfo_JSON.json", "w") as outfile:
                json.dump(r_dict, outfile)

            plotconfigs["Rule " + str(rule)] = r_list

        for nuni_rule in non_uni_rules.keys():
            nuni_rule_rs = non_uni_rules.get(nuni_rule)
            r_dict = {}
            r_list = []
            for rci_value in RCI_values:
                R = rci_value[0]
                rule_scheme = nuni_rule_rs.get(R)
                non_uni_size = len(set(rule_scheme))
                data_interpreter = self.open_data_interpreter("5bit", distractor_period=10)
                with multiprocessing.Pool(threads) as p:
                    results = p.starmap(run_non_uniform_five_bit,
                                        [(data_interpreter, rci_value, "perceptron_sgd", rule_scheme) for _ in
                                         range(number_of_tests)])

                r_dict["R=" + str(rci_value[0])] = results
                r_list.append(int(np.mean(results)))
            with open(file_location + "/../experiment_data/rule_testing/nuni_rule(" + str(non_uni_size) + ")_allinfo_JSON.json",
                      "w") as outfile:
                json.dump(r_dict, outfile)

            plotconfigs["Non uniform(" + str(non_uni_size) + " rules)"] = r_list


        print(plotconfigs)
        with open(file_location + "/../experiment_data/rule_testing/full_plotconfig.json", "w") as outfile:
            json.dump(plotconfigs, outfile)


        self.create_graph_from_jsonconfig(file_location + "/../experiment_data/rule_testing/full_plotconfig.json", Rs)


    def evolve_non_uniform_ca(self):
        pass

    def classifier_testing(self):
        classifiers = ["linear-svm", "perceptron_sgd"]
        RCI_values_some = [(4,4,4),(4,10,2)]
        RCI_values_r_change = [(x,5,2) for x in [2,4,6,8]]


        RCI_values = RCI_values_r_change
        #RCI_values = [(1,1,1)]
        distractor_periods = [10, 50, 100, 200]
        distractor_periods = [10, 25, 50]
        threads = 3
        number_of_tests = threads*10

        plotconfigs = {}
        plotlabels = []


        for rci_value in RCI_values:
            classifier_dict = {}
            for classifier in classifiers:
                plotlabels.append(classifier)
                distactor_dict = {}
                for distractor_period in distractor_periods:
                    data_interpreter = self.open_data_interpreter("5bit", distractor_period)
                    with multiprocessing.Pool(threads) as p:
                        results = p.starmap(run_five_bit, [(data_interpreter, rci_value, classifier) for _ in range(number_of_tests)])

                    distactor_dict[distractor_period] = int(np.mean(results))
                classifier_dict[classifier] = distactor_dict
            plotconfigs["R=" + str(rci_value[0])] = classifier_dict

        print(plotconfigs)
        self.create_graph_from_plotconfig(plotconfigs, plotlabels)

    def create_graph_from_plotconfig(self, plotconfig, plotlabels):
        for r_value in plotconfig.keys():
            name = str(r_value)
            plots = []
            for classifier in plotconfig.get(r_value).keys():
                distractor_periods = plotconfig.get(r_value).get(classifier).keys()
                distractor_periods = sorted(distractor_periods, reverse=True)  # ascending
                sucess_rates = []
                for distractor_period in distractor_periods:
                    sucess_rates.append(plotconfig.get(r_value).get(classifier).get(distractor_period))
                plots.append((distractor_periods, sucess_rates))

            # Make an example plot with two subplots...
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            clf_plot1 = ax1.plot(plots[0][0], plots[0][1], 'rs', label=plotlabels[0])
            ax1.plot(plots[0][0], plots[0][1], 'r--')
            clf_plot2 = ax1.plot(plots[1][0], plots[1][1], 'bs', label=plotlabels[1])
            ax1.plot(plots[1][0], plots[1][1], 'b--')
            ax1.set_ylim([-10,1010])
            ax1.set_xlim([0,53])

            #ax2 = fig.add_subplot(2,1,2)
            #ax2.plot(plots[0][0], plots[0][1], 'rs', plots[1][0], plots[1][1], 'bs')
            plt.ylabel("classifiertest")

            legend = ax1.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            # Save the full figure...
            file_location = os.path.dirname(os.path.realpath(__file__))
            fig.savefig(file_location+"/../experiment_data/clf_test/" + name)

    def create_graph_from_jsonconfig(self, json_file_location, Rs = (2, 4, 6, 8)):
        plot_data = None
        with open(json_file_location) as data_file:
            plot_data = json.load(data_file)

        fig = plt.figure(figsize=(10, 8))

        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylim([-10, 1550])
        ax1.set_xlim([1, 9])
        ax1.set_title("Rule testing on 5-bit problem with T_d=10")
        name = str("Full plot")
        plot_colors = ["r", "g", "b", "y", "m", "k", "w", "c"]
        i = 0
        for rule in sorted(plot_data.keys(), key=len):  # For nice legend
            plots = []

            # Make an example plot with two subplots...
            clf_plot1 = ax1.plot(Rs, plot_data.get(rule), plot_colors[i%len(plot_colors)] + 's', label=str(rule))
            ax1.plot(Rs, plot_data.get(rule), plot_colors[i%len(plot_colors)] + '--')

            plt.xlabel("R-values")
            plt.ylabel("Permille(1/1000) correct")

            legend = ax1.legend(loc='upper left', shadow=True, prop={'size': 12})
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            # Save the full figure...
            file_location = os.path.dirname(os.path.realpath(__file__))
            fig.savefig(file_location + "/../experiment_data/rule_testing/" + name)
            i+=1






