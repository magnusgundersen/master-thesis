from ea import evoalg as evoalg
from ea import individual as ind
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


import random
# testing
class NonUniCAGenotype(ind.Genotype):
    def __init__(self, parent_genotype_one, parent_genotype_two):
        super().__init__(parent_genotype_one, parent_genotype_two)

    def init_first_genotype(self):
        #print("First genotype!")
        number_of_rules = 8
        self.rule_scheme = [random.choice([0,1]) for _ in range(8*8)]

    def get_representation(self):
        pass

    def reproduce(self, other_genotype, crossover_rate=0.4, mutation_rate=0.01):
        self.rule_scheme = self.rule_scheme[:len(self.rule_scheme)*crossover_rate] + \
                           other_genotype.rule_scheme[len(self.rule_scheme)*crossover_rate:]
        random_number = random.random()
        if random_number<mutation_rate:
            bitflip = random.randint(0,len(self.rule_scheme)-1)  # last number is included.
            self.rule_scheme[bitflip] = 0 if self.rule_scheme[bitflip] == 1 else 1

class NonUniCAPhenotype(ind.Phenotype):
    def __init__(self, genotype, ca_size):
        super().__init__(genotype)
        self.ca_size = ca_size
        self.non_uniform_config = []
        self.develop_from_genotype()

    def develop_from_genotype(self):
        binary_rule_scheme = self.genotype.rule_scheme
        bit_encoding = 8
        number_of_rules_in_rule_scheme = len(binary_rule_scheme) // bit_encoding  # n bit encoding of rules
        list_of_rules = []
        for i in range(number_of_rules_in_rule_scheme):
            rule = binary_rule_scheme[i*bit_encoding:(i+1)*bit_encoding]
            rule = map(str, rule)
            rule = "".join(rule)
            rule = int(rule,2)
            list_of_rules.append(rule)

        for i in range(self.ca_size):
            self.non_uniform_config.append(list_of_rules[i%number_of_rules_in_rule_scheme])


        #print(self.non_uniform_config)



class NonUniCAIndividual(ind.Individual):
    def __init__(self, parent_genotype_one=None, parent_genotype_two=None):
        super().__init__(parent_genotype_one, parent_genotype_two)
        self.genotype = NonUniCAGenotype(parent_genotype_one, parent_genotype_two)
        self.phenotype = None



    def develop(self):
        self.phenotype = NonUniCAPhenotype(self.genotype, 32)

    def reproduce(self, other_parent_genotype):
        child = NonUniCAIndividual(other_parent_genotype)
        return child

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

class NonUniCAProblem(evoalg.EAProblem):
    def __init__(self):
        super().__init__()
        self.max_number_of_generations = 15

    def test_fitness(self, individual):


        data_interpreter = self.open_data_interpreter("5bit")
        reCA_problem = reCA.ReCAProblem(data_interpreter)
        reCA_config = reCA.ReCAConfig()

        reCA_rule_scheme = reCA.ReCAruleConfig(individual.phenotype.non_uniform_config)

        reCA_config.set_non_uniform_config(reCA_rule_scheme, R=4, C=2, I=1)
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_problem()
        print(str(reCA_out.total_correct) + " of " + str(len(reCA_out.all_test_examples)))

        fitness = int((reCA_out.total_correct/len(reCA_out.all_test_examples))*1000)
        fitness = 1 if fitness == 0 else fitness
        individual.fitness = fitness
        print("Tested individual: " + str(individual))

        return individual.fitness


    def get_initial_population(self):
        initi_pop_size = 6
        return [NonUniCAIndividual() for _ in range(initi_pop_size)]

    def open_data_interpreter(self, type_of_interpreter):
        if type_of_interpreter == "europarl":
            return data_int.TranslationBuilder()

        elif type_of_interpreter == "5bit":
            return data_int.FiveBitBuilder()

        elif type_of_interpreter == "20bit":
            return data_int.TwentyBitBuilder()


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

if __name__ == "__main__":
    nonUniCAprob = NonUniCAProblem()
    ea = evoalg.EA()

    best_ind = ea.solve(nonUniCAprob)
    print("best individual: " + str(best_ind))
    print(best_ind.phenotype.non_uniform_config)


    data_interpreter = open_data_interpreter("5bit")
    reCA_problem = reCA.ReCAProblem(data_interpreter)
    reCA_config = reCA.ReCAConfig()

    reCA_rule_scheme = reCA.ReCAruleConfig(best_ind.phenotype.non_uniform_config)

    reCA_config.set_non_uniform_config(reCA_rule_scheme, R=4, C=2, I=1)
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



