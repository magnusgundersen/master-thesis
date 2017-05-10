import json
import matplotlib.pyplot as plt
import os
import gui.ca_basic_visualizer as bvis

def create_graph_from_plotconfig(plotconfig, plotlabels):
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
        ax1 = fig.add_subplot(1, 1, 1)
        clf_plot1 = ax1.plot(plots[0][0], plots[0][1], 'rs', label=plotlabels[0])
        ax1.plot(plots[0][0], plots[0][1], 'r--')
        clf_plot2 = ax1.plot(plots[1][0], plots[1][1], 'bs', label=plotlabels[1])
        ax1.plot(plots[1][0], plots[1][1], 'b--')
        ax1.set_ylim([-10, 1010])
        ax1.set_xlim([0, 53])

        # ax2 = fig.add_subplot(2,1,2)
        # ax2.plot(plots[0][0], plots[0][1], 'rs', plots[1][0], plots[1][1], 'bs')
        plt.ylabel("classifiertest")

        legend = ax1.legend(loc='upper center', shadow=True)
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Save the full figure...
        file_location = os.path.dirname(os.path.realpath(__file__))
        fig.savefig(file_location + "/../experiment_data/clf_test/" + name)


def create_graph_from_jsonconfig(json_file_location, Rs=(600)):
    plot_data = None
    with open(json_file_location) as data_file:
        plot_data = json.load(data_file)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim([-10, 1450])
    ax1.set_xlim([95, 105])
    ax1.set_title("Rule testing on 5-bit problem with T_d=10")
    name = str("Full plot")
    plot_colors = ["r", "g", "b", "y", "m", "k", "w", "c"]
    i = 0
    for rule in sorted(plot_data.keys(), key=len):  # For nice legend
        plots = []

        # Make an example plot with two subplots...
        clf_plot1 = ax1.plot(Rs, plot_data.get(rule), plot_colors[i % len(plot_colors)] + 's', label=str(rule))
        ax1.plot(Rs, plot_data.get(rule), plot_colors[i % len(plot_colors)] + '--')

        plt.xlabel("R-values")
        plt.ylabel("Permille(1/1000) correct")

        legend = ax1.legend(loc='upper left', shadow=True, prop={'size': 12})
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Save the full figure...
        file_location = os.path.dirname(os.path.realpath(__file__))
        fig.savefig(file_location + "/../experiment_data/rule_testing/" + name)
        i += 1

def visualize_example_run(run_config):
    whole_output = []
    lists_of_states = [output.list_of_states for output in run_config]
    for output in lists_of_states:
        width = len(output[0])
        new_output = []
        for line in output:
            new_output.append([(-1 if i == 0 else 1) for i in line])

        whole_output.extend(new_output)
        whole_output.extend([[0 for _ in range(width)]])
    bvis.visualize(whole_output)

def visualize_ca_run(ca_states):
    bvis.visualize(ca_states)


def make_fitnessgraph(ea_output, name):
    fitness_list = [ind.fitness for ind in ea_output.best_individuals_per_gen]
    mean_fitness_list = ea_output.mean_fitness_per_gen
    std_fitness_list = ea_output.std_per_gen
    #plt.plot(ea_output.mean_fitness_per_gen)
    #plt.plot(ea_output.std_per_gen)

    #plt.xlabel('Fitnessplot: ' + name)
    file_location = os.path.dirname(os.path.realpath(__file__))
    #plt.savefig(file_location+"/../experiment_data/ea_runs/" + name)
    #plt.close()

    plot_data = [fitness_list, mean_fitness_list, std_fitness_list]
    names = ["Best fitness", "Mean fitness", "Std. fitness"]
    generations = len(fitness_list)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_ylim([-10, 1250])
    ax1.set_xlim([-1, generations+3])
    ax1.set_title("Fitness plot: " + name[:-9])
    name = name
    plot_colors = ["r", "g", "b", "y", "m", "k", "w", "c"]
    i = 0

    for data in plot_data:  # For nice legend
        plots = []

        # Make an example plot with two subplots...
        clf_plot1 = ax1.plot(range(generations), data, label=str(names[i]))
        #ax1.plot(range(generations), data, plot_colors[i % len(plot_colors)] + '--')  plot_colors[i % len(plot_colors)] + 's'

        plt.xlabel("Generations")
        plt.ylabel("Permille(1/1000) correct")

        legend = ax1.legend(loc='upper left', shadow=True, prop={'size': 12})
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Save the full figure...
        file_location = os.path.dirname(os.path.realpath(__file__))
        fig.savefig(file_location + "/../experiment_data/ea_runs/" + name)
        i += 1
