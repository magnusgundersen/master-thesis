from ea import evoalg as evoalg
from ea import individual as ind
from reca import reca_system as reCA
from master import project as p
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
import multiprocessing


# Workers:
def fitness_test_worker(individual, R=100, C=1, I=4, classifier="perceptron_sgd", time_transition="xor",
                        distractor_period=10, train_ex=32, test_ex=32, tests_per_ind=6):
    """
        Method for running the develop_and_test with multiprocessing
        :param individual:
        :param problem:
        :return:
        """

    fitness = []
    for _ in range(tests_per_ind):
        reCA_problem = reCA.ReCAProblem(
            p.open_data_interpreter("5bit", distractor_period=distractor_period, training_ex=train_ex, testing_ex=test_ex))
        reCA_config = reCA.ReCAConfig()
        reCA_rule_scheme = reCA.ReCAruleConfig(non_uniform_list=individual.phenotype.non_uniform_config)
        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule_scheme, R=R, C=C, I=I,
                                              classifier=classifier,
                                              time_transition=time_transition)
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_problem()
        fitness.append(int((reCA_out.total_correct / len(reCA_out.all_test_examples)) * 1000))


    fitness_std = int(np.std(fitness))
    fitness = int(np.mean(fitness))

    # fitness = fitness if (fitness<850) else fitness-fitness_std*(1000/fitness)
    fitness = 1 if fitness == 0 else fitness  # avoid div by zero
    individual.fitness = fitness
    individual.fitness_std = fitness_std

    return individual



# Classes:
class NonUniCAGenotype(ind.Genotype):
    def __init__(self, parent_genotype_one, parent_genotype_two, allowed_number_of_rules=4):
        self.rule_scheme = []
        self.allowed_number_of_rules = allowed_number_of_rules
        self.bit_per_rule = 8
        self.rule_bound_crossover = True
        self.crossover_type = "two point"

        super().__init__(parent_genotype_one, parent_genotype_two)

    def init_first_genotype(self):
        self.rule_scheme = [random.choice([0,1]) for _ in range(self.allowed_number_of_rules*self.bit_per_rule)]


    def reproduce(self, parent_one_genotype, parent_two_genotype, crossover_rate=random.random(), mutation_rate=0.15):

        # Crossover
        number_of_crossover_points = len(parent_one_genotype.rule_scheme) // self.bit_per_rule
        if self.crossover_type == "single point":
            if self.rule_bound_crossover:
                crossover_point = int(number_of_crossover_points*crossover_rate)
                crossover_point *= self.bit_per_rule

            else:
                crossover_point = len(parent_one_genotype.rule_scheme)*crossover_rate

            self.rule_scheme = parent_one_genotype.rule_scheme[:crossover_point] + \
                               parent_two_genotype.rule_scheme[crossover_point:]

        elif self.crossover_type == "two point":
            if self.rule_bound_crossover:
                crossover_size = int(number_of_crossover_points * crossover_rate)  # int floors
                point_one = random.randint(0, int(number_of_crossover_points - crossover_size))
                point_two = point_one + crossover_size

            else:
                crossover_size = len(parent_one_genotype.rule_scheme)*crossover_rate # int floors
                point_one = random.randint(0, int(parent_one_genotype.rule_scheme - crossover_size))
                point_two = point_one + crossover_size

            self.rule_scheme = parent_one_genotype.rule_scheme[:point_one] + \
                               parent_two_genotype.rule_scheme[point_one:point_two] + \
                               parent_one_genotype.rule_scheme[point_two:]

        # Mutation
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
        self.fitness_std = 0



    def develop(self, ca_size=0):
        if ca_size == 0:
            ca_size = self.ca_size
        if self.phenotype == None or ca_size != self.ca_size:
            self.phenotype = NonUniCAPhenotype(self.genotype, ca_size)

    def reproduce(self, other_parent_genotype):
        child = NonUniCAIndividual(self.allowed_number_of_rules, self.ca_size,
                                   parent_genotype_one=self.genotype, parent_genotype_two=other_parent_genotype)
        return child

    def serialize(self):
        # Genotype:
        binary_base_rule = self.genotype.rule_scheme
        binary_base_list = [[binary_base_rule[self.genotype.bit_per_rule*i+j] for j in range(self.genotype.bit_per_rule)] for i in range(len(binary_base_rule)//self.genotype.bit_per_rule)]
        int_base_rule = [int("".join([str(bin_int) for bin_int in bin_rule]),2) for bin_rule in binary_base_list]

        #Phenotype

    def __str__(self):
        return "NuniRule"+ str(random.randint(0,10000)) + "_f=" + str(self.fitness)



class NonUniCAProblem(evoalg.EAProblem):
    def __init__(self, ca_config=None, init_pop_size=40, allowed_number_of_rules=4, fitness_threshold=900, max_number_of_generations=2, test_per_ind=4):
        super().__init__()
        self.max_number_of_generations = max_number_of_generations
        self.N = ca_config.get("N")
        self.R = ca_config.get("R")
        self.C = ca_config.get("C")
        self.I = ca_config.get("I")
        self.test_per_ind = test_per_ind
        self.init_pop_size = init_pop_size
        self.fitness_threshold_value = fitness_threshold
        self.allowed_number_of_rules = allowed_number_of_rules
        self.ca_size = self.N*self.C*self.R
        self.name = "Non uniform CA"+"(" + str(self.N) + "*" + str(self.C) + "*" + str(self.R) + "): PopSize: " + str(init_pop_size) + " Max gens: " + str(max_number_of_generations)


    #staticmethod
    def test_fitness(self, individual):
        return fitness_test_worker(individual, R=self.R, C=self.C, I=self.I)

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


    def fitness_threshold(self, *args):
        fitness = args[0]
        if fitness >= self.fitness_threshold_value:
            return True
        else:
            return False

    def __str__(self):
        return self.name






