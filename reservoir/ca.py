__author__ = 'magnus'
import pprint

class ElemCAReservoir:
    """
    Elem. CA reservoir.
    Elem. meaning that the cell can only have values 0 or 1, and has 3 neighbors.

    The reservoir has elementary rules governing all the cells, given by a list of number 0-255, one for each cell.

    """
    def __init__(self):
        self.non_uniform = False
        self.rules = []  # List of the same width as the CA. Non-uniform CA.


    def set_uniform_rule(self, rule_number):
        """
        Short function to init. a uniform CA.
        :param rule_number:
        :return:
        """
        self.non_uniform = False
        rule = Rule(rule_number)
        self.rules.append(rule)

    def set_rules(self, rule_list):
        """

        :param rule_list:
        :return:
        """
        self.non_uniform = True
        for rule in rule_list:
            self.rules.append(Rule(rule))



    @staticmethod
    def run_simulation_step(ca_vector, rules):
        length = len(ca_vector)
        next_ca_vector = []
        if length != len(rules):
            raise ValueError("[CA simulation] Not correct number of rules: "
                             "Should be "+str(length)+" but was " + str(len(rules)))
        #Wrap around
        for i in range(length):
            left_index = (i-1) % length
            mid_index = i
            right_index = (i+1) % length
            rule_at_i = rules[i]
            next_ca_vector.append(rule_at_i.getOutput([ca_vector[left_index],
                                  ca_vector[mid_index], ca_vector[right_index]]))
        return next_ca_vector


    def run_simulation(self, initial_inputs, iterations):
        """
        Runs a simulation of the initial input, for a given iterations


        Returns the whole list of generations
        :param initial_inputs:
        :param iterations:
        :return:
        """
        all_generations = [initial_inputs]
        current_generation = all_generations[0]

        if not self.non_uniform and (len(self.rules) == 1):  # generation of same rule on the fly
            for _ in range(len(initial_inputs) - 1):
                self.rules.append(self.rules[0])

        for i in range(iterations): # Run for I iterations
            current_generation = self.run_simulation_step(current_generation, self.rules)
            all_generations.append(current_generation)
        return all_generations



class Rule:
    def __init__(self, number=0):
        """

        :param number: Corresponds to Wolfram number of elem. CA rules
        :return:
        """
        self.number = number

    def getRuleScheme(self, rule_number):
        """
        :param rule_number:
        :return:
        """
        binrule = format(rule_number, "08b")  # convert to binary, with fill of zeros until the string is of length 8

        rule = {
            (1,1,1): int(binrule[0]),
            (1,1,0): int(binrule[1]),
            (1,0,1): int(binrule[2]),
            (1,0,0): int(binrule[3]),
            (0,1,1): int(binrule[4]),
            (0,1,0): int(binrule[5]),
            (0,0,1): int(binrule[6]),
            (0,0,0): int(binrule[7])
        }

        return rule

    def getOutput(self, input_array):
        if len(input_array) != 3:
            raise ValueError

        scheme = self.getRuleScheme(self.number)
        output = scheme[(input_array[0], input_array[1], input_array[2])]

        return output


