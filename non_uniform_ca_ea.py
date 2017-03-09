from ea import evoalg as evoalg
from ea import individual as ind
from reca import reca_system as reCA

import numpy as np
from gui import ca_basic_visualizer as bviz
import random
import pprint
import itertools # for permutations
import csv
import os
import pickle as pickle
import time
import experiment_data.data_interpreter as data_int
import reservoir.ca as ca
import sys
import signal
import pickle
import matplotlib.pyplot as plt
import json
import datetime


import random
# testing
class NonUniCAGenotype(ind.Genotype):
    def __init__(self, parent_genotype_one, parent_genotype_two, allowed_number_of_rules=4):
        self.rule_scheme = []
        self.allowed_number_of_rules = allowed_number_of_rules
        self.bit_per_rule = 8
        super().__init__(parent_genotype_one, parent_genotype_two)

    def init_first_genotype(self):
        self.rule_scheme = [random.choice([0,1]) for _ in range(self.allowed_number_of_rules*self.bit_per_rule)]

    def get_representation(self):
        pass

    def reproduce(self, parent_one_genotype, parent_two_genotype, crossover_rate=0.5, mutation_rate=0.05):
        rule_bound_crossover = True

        if rule_bound_crossover:
            number_of_crossover_points = len(parent_one_genotype.rule_scheme) % self.bit_per_rule
            crossover_point = int(number_of_crossover_points*crossover_rate)
            crossover_point *= self.bit_per_rule

        else:
            crossover_point = len(parent_one_genotype.rule_scheme)*crossover_rate

        self.rule_scheme = parent_one_genotype.rule_scheme[:crossover_point] + \
                           parent_two_genotype.rule_scheme[crossover_point:]
        random_number = random.random()

        if random_number < mutation_rate:
            number_of_flips = random.choice([1,1,1,1,1,1,2,2,3])
            for _ in range(number_of_flips):
                bitflip = random.randint(0,len(self.rule_scheme)-1)  # last number is included.
                self.rule_scheme[bitflip] = 0 if self.rule_scheme[bitflip] == 1 else 1


class NonUniCAPhenotype(ind.Phenotype):
    def __init__(self, genotype, ca_size):
        super().__init__(genotype)
        self.ca_size = ca_size
        self.non_uniform_config = []
        self.rule_blocks = True
        self.develop_from_genotype()

    def develop_from_genotype(self):
        binary_rule_scheme = self.genotype.rule_scheme
        bit_encoding = 8
        number_of_rules_in_rule_scheme = len(binary_rule_scheme) // bit_encoding  # n bit encoding of rules

        if self.rule_blocks:
            cells_per_rule, rest = divmod(self.ca_size, number_of_rules_in_rule_scheme)
            # The rest-cell(s) will have the same rule as the last cell


            list_of_rules = []
            for i in range(number_of_rules_in_rule_scheme):
                rule = binary_rule_scheme[i * bit_encoding:(i + 1) * bit_encoding]
                rule = map(str, rule)
                rule = "".join(rule)
                rule = int(rule, 2)
                list_of_rules.append(rule)

            for rule in list_of_rules:
                for _ in range(cells_per_rule):
                    self.non_uniform_config.append(rule)
            for _ in range(rest):
                self.non_uniform_config.append(list_of_rules[-1])  # Last rule

        else:
            list_of_rules = []
            for i in range(number_of_rules_in_rule_scheme):
                rule = binary_rule_scheme[i*bit_encoding:(i+1)*bit_encoding]
                rule = map(str, rule)
                rule = "".join(rule)
                rule = int(rule,2)
                list_of_rules.append(rule)

            for i in range(self.ca_size):
                self.non_uniform_config.append(list_of_rules[i%number_of_rules_in_rule_scheme])




class NonUniCAIndividual(ind.Individual):
    def __init__(self, allowed_number_of_rules=4, ca_size=96, parent_genotype_one=None, parent_genotype_two=None, ):
        super().__init__(parent_genotype_one, parent_genotype_two)
        self.genotype = NonUniCAGenotype(parent_genotype_one, parent_genotype_two, allowed_number_of_rules)
        self.phenotype = None
        self.ca_size = ca_size
        self.allowed_number_of_rules=allowed_number_of_rules


    def develop(self):
        self.phenotype = NonUniCAPhenotype(self.genotype, self.ca_size)

    def reproduce(self, other_parent_genotype):
        child = NonUniCAIndividual(self.allowed_number_of_rules, self.ca_size,
                                   parent_genotype_one=self.genotype, parent_genotype_two=other_parent_genotype)
        return child


class NonUniCAProblem(evoalg.EAProblem):
    def __init__(self, init_pop_size=40, ca_size=40, allowed_number_of_rules=4, fitness_threshold=900, max_number_of_generations=2, R=6, C=4, I=4, test_per_ind=4):
        super().__init__()
        self.max_number_of_generations = max_number_of_generations
        self.R = R
        self.C = C
        self.I = I
        self.test_per_ind = test_per_ind
        self.init_pop_size = init_pop_size
        self.fitness_threshold_value = fitness_threshold
        self.allowed_number_of_rules = allowed_number_of_rules
        self.ca_size = ca_size

    def test_fitness(self, individual):
        fitness = 0
        for _ in range(self.test_per_ind):
            data_interpreter = self.open_data_interpreter("5bit")
            reCA_problem = reCA.ReCAProblem(data_interpreter)
            reCA_config = reCA.ReCAConfig()

            reCA_rule_scheme = reCA.ReCAruleConfig(individual.phenotype.non_uniform_config)

            reCA_config.set_non_uniform_config(reCA_rule_scheme, R=self.R, C=self.C, I=self.I)
            reCA_system = reCA.ReCASystem()

            reCA_system.set_problem(reCA_problem)
            reCA_system.set_config(reCA_config)
            reCA_system.initialize_rc()
            reCA_system.tackle_ReCA_problem()

            reCA_out = reCA_system.test_on_problem()
            fitness += int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)

        #pseudo_lambda = self.calculate_pseudo_lambda(individual.phenotype.non_uniform_config)
        #fitness += pseudo_lambda
        fitness = 1 if fitness == 0 else fitness//self.test_per_ind
        individual.fitness = fitness
        return individual.fitness

    def calculate_pseudo_lambda(self, rule_set):
        ca_simulator = ca.ElemCAReservoir()
        ca_simulator.set_rules(rule_set)
        initial_input = np.zeros(len(rule_set))
        initial_input[int(len(initial_input)/2)] = 1
        simulation = ca_simulator.run_simulation(initial_input, 1)
        second_row = simulation[1]
        ones = list(second_row).count(1)
        return int(1000/(ones+1))


    def get_initial_population(self):
        return [NonUniCAIndividual(ca_size=self.ca_size, allowed_number_of_rules=self.allowed_number_of_rules) for _ in range(self.init_pop_size)]

    def open_data_interpreter(self, type_of_interpreter):
        if type_of_interpreter == "europarl":
            return data_int.TranslationBuilder()

        elif type_of_interpreter == "5bit":
            return data_int.FiveBitBuilder()

        elif type_of_interpreter == "20bit":
            return data_int.TwentyBitBuilder()

    def fitness_threshold(self, *args):
        fitness = args[0]
        if fitness >= self.fitness_threshold_value:
            return True
        else:
            return False


def open_data_interpreter(type_of_interpreter):
    if type_of_interpreter == "europarl":
        return data_int.TranslationBuilder()

    elif type_of_interpreter == "5bit":
        return data_int.FiveBitBuilder()

    elif type_of_interpreter == "20bit":
        return data_int.TwentyBitBuilder()


def visualise_example(training_array):
    visualizer = bviz.CAVisualizer()
    visualizer.visualize(training_array)
def viz(rule_scheme, R, C, I):
    data_interpreter = open_data_interpreter("5bit")
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()

    reCA_rule_scheme = reCA.ReCAruleConfig(rule_scheme)

    reCA_config.set_non_uniform_config(reCA_rule_scheme, R=R, C=C, I=I)
    reCA_system = reCA.ReCASystem()

    reCA_system.set_problem(reCA_problem)
    reCA_system.set_config(reCA_config)
    reCA_system.initialize_rc()
    reCA_system.tackle_ReCA_problem()

    reCA_out = reCA_system.test_on_problem()
    print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))

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
    visualise_example(whole_output)

def make_fitnessgraph(ea_output, name):
    plt.plot([ind.fitness for ind in ea_output.best_individuals_per_gen])
    #plt.plot(ea_output.mean_fitness_per_gen)
    #plt.plot(ea_output.std_per_gen)
    plt.xlabel('Fitnessplot: ' + name)
    plt.savefig("experiment_data/ea_runs/" + name)
    plt.close()

def run_ea_0903_test():
    C = 1
    Rs = [1, 2]
    I = 1
    N = 4  # For 5-bit task. DO NOT CHANGE

    file_location = os.path.dirname(os.path.realpath(__file__))
    pop_size = 7
    max_no_generations = 1
    tests_per_individual = 1
    number_of_rules_list = [1, 2, 3, 4]
    print_est = True
    before = time.time()
    for number_of_rules in number_of_rules_list:
        for R in Rs:
            ca_size = C * R * N
            nonUniCAprob = NonUniCAProblem(R=R, I=I, C=C, fitness_threshold=1000, init_pop_size=pop_size,
                                           max_number_of_generations=max_no_generations,
                                           allowed_number_of_rules=number_of_rules, ca_size=ca_size, test_per_ind=tests_per_individual)
            ea = evoalg.EA()

            ea_output = ea.solve(nonUniCAprob)

            # pickle.dump(ea_output, open("ea.pkl", "wb"))
            run_name = "earun_R" + str(R) + "C" + str(C) + "I" + str(I) + \
                       "_rules" + str(number_of_rules) + "_popsize" + str(pop_size) + \
                       "_gens" + str(max_no_generations)

            pickle.dump(ea_output, open(file_location+"/experiment_data/ea_runs/" + run_name + ".pkl", "wb"))
            best_individual = ea_output.best_individual
            best_individ_scheme = best_individual.phenotype.non_uniform_config
            non_uni_rule_serialize = {}
            non_uni_rule_serialize["full_size_rule_list"] = best_individ_scheme
            non_uni_rule_serialize["raw rule"] = best_individual.genotype.rule_scheme
            ea_data = {"R": R, "C": C, "I": I, "N": N,
                       "ca size": ca_size,
                       "popsize":pop_size,
                       "max_gens":max_no_generations,
                       "test per ind": tests_per_individual,
                       "allowed number of rules": number_of_rules,
                       }

            non_uni_rule_serialize["ea_data"] = ea_data

            with open(file_location+"/experiment_data/ea_runs/" + run_name +"JSON.json", "w") as outfile:
                json.dump(non_uni_rule_serialize, outfile)

            make_fitnessgraph(ea_output, run_name)
        if print_est:
            ts = time.time()
            print("Time now : " +str(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
            print("Time used: " + str(ts-before))
            print("Total time est.: " + str((ts-before)*(len(number_of_rules_list))))
            print("Time est. left : " + str((ts-before)*(len(number_of_rules_list)-1)))
            print_est = False

    print("Actual time usage: " + str(time.time()-before))

if __name__ == "__main__":
    run_ea_0903_test()
    """
    C = 5
    R = 2
    I = 2
    N = 4  # For 5-bit task. DO NOT CHANGE
    pop_size = 8
    max_no_generations = 2
    tests_per_individual = 1
    ca_size = C*R*N
    number_of_rules = 3




    nonUniCAprob = NonUniCAProblem(R=R, I=I, C=C, fitness_threshold=1000, init_pop_size=pop_size,
                                   max_number_of_generations=max_no_generations, allowed_number_of_rules=number_of_rules, ca_size=ca_size)
    ea = evoalg.EA()

    ea_output = ea.solve(nonUniCAprob)

    #pickle.dump(ea_output, open("ea.pkl", "wb"))
    run_name = "earun_R"+str(R)+"C"+str(C)+"I"+str(I)+ \
                                "_rules" + str(number_of_rules)+"_popsize" + str(pop_size) + \
                                "_gens" + str(max_no_generations)

    pickle.dump(ea_output, open("experiment_data/ea_runs/"+run_name+".pkl", "wb"))
    best_individ_scheme = ea_output.best_individual.phenotype.non_uniform_config
    with open("experiment_data/ea_runs/"+run_name+".txt", "w+") as f:
        f.write(str(best_individ_scheme))

    make_fitnessgraph(ea_output, run_name)

    #print(best_individ_scheme)


    #viz(best_individ_scheme, R, C, I)
    """





