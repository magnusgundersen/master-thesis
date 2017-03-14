__author__ = 'magnus'
import pprint
import numpy as np
import multiprocessing

def ca_step(i, length, rules, ca_vector):
    left_index = (i - 1) % length
    mid_index = i
    right_index = (i + 1) % length
    rule_at_i = rules[i]
    return rule_at_i.getOutput([ca_vector[left_index], ca_vector[mid_index], ca_vector[right_index]])

class ElemCAReservoir:
    """
    Elem. CA reservoir.
    Elem. meaning that the cell can only have values 0 or 1, and has 3 neighbors.

    The reservoir has elementary rules governing all the cells, given by a list of number 0-255, one for each cell.

    """
    def __init__(self, ca_size=0):
        self.non_uniform = False
        self.rules = []  # List of the same width as the CA. Non-uniform CA.
        self.birth = lambda x, y ,z :(
            ((x[10] == 0) & (y[10] == 0) & (z[10] == 0)) |
            ((x[10] == 0) & (y[10] == 0) & (z[10] == 1))
                                    )
        self.birth = lambda x, y, z: (
            (
                (True) |
                ((x[10] == 0) & (y[10] == 0) & (z[10] == 1))
            )
        )

        self.survive = lambda x, y ,z :(((x[:] == 1) & (y[:] == 1) & (z[:] == 0)) | (
            (x[:] == 0) & (y[:] == 1) & (z[:] == 1)))
        self.ca_size = ca_size

    def build_logical_expressions(self, rule_scheme):
        #rule_scheme = [10]+[150 for _ in range(5)]+[77]+[90 for _ in range(9)] + [110 for _ in range(len(rule_scheme))]

        self.rule_intervals = {}
        start = 0
        end = 1
        current_rule = rule_scheme[0]
        for i in range(len(rule_scheme[1:])):
            new_rule = rule_scheme[1:][i]
            if new_rule == current_rule:
                end += 1
            elif current_rule != new_rule:
                self.rule_intervals[(start, end)] = current_rule
                current_rule = new_rule
                start = i + 1
                end = end + 1
            if i == (len(rule_scheme[1:])-1):
                self.rule_intervals[(start, end)] = current_rule

        for interval in self.rule_intervals.keys():
            rule = self.rule_intervals.get(interval)
            rule = Rule(rule)
            self.rule_intervals[interval] = rule.get_logical_expression()


    def set_rule_config(self, rule_config):
        """
        Configures the reservoir with the given config.

        :param rule_scheme:
        :return:
        """
        self.rule_config = rule_config
        rule_scheme = rule_config.get_scheme(self.ca_size)
        self.build_logical_expressions(rule_scheme)




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
        next_ca_vector = np.zeros(length, dtype="uint8")
        if length != len(rules):
            raise ValueError("[CA simulation] Not correct number of rules: "
                             "Should be "+str(length)+" but was " + str(len(rules)))

        parallel = False
        if parallel:
            with multiprocessing.Pool(7) as p:
                list_next_ca_vector = p.starmap(ca_step, [(i, length, rules, ca_vector) for i in range(length)])
            next_ca_vector = np.array(list_next_ca_vector, dtype="uint8")

        else:
            #Wrap around
            for i in range(length):
                left_index = (i-1) % length
                mid_index = i
                right_index = (i+1) % length
                rule_at_i = rules[i]
                next_ca_vector[i]=(rule_at_i.getOutput([ca_vector[left_index],
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
        return self.run_simulation_np(initial_inputs, iterations)
        all_generations = [initial_inputs]
        current_generation = all_generations[0]

        if not self.non_uniform and (len(self.rules) == 1):  # generation of same rule on the fly
            for _ in range(len(initial_inputs) - 1):
                self.rules.append(self.rules[0])


        for i in range(iterations): # Run for I iterations
            current_generation = self.run_simulation_step(current_generation, self.rules)
            all_generations.append(current_generation)
        return all_generations

    def run_simulation_np(self, initial_inputs, iterations):
        """
        Experimental method with using numpy
        :param initial_inputs:
        :param iterations:
        :return:
        """
        #print(initial_inputs)
        output = [initial_inputs]
        current_input = np.array(initial_inputs, dtype="uint8")
        for i in range(iterations):
            Z = current_input
            list_of_lefts = np.roll(Z, 1)
            list_of_rights = np.roll(Z, -1)
            birth = []
            survive = []
            #print(str(self.rule_intervals.keys()))
            for interval_start, interval_end in self.rule_intervals.keys():
                birth_function, survive_function = self.rule_intervals.get((interval_start, interval_end))
                birth += list(birth_function(list_of_lefts[interval_start:interval_end],
                                        Z[interval_start:interval_end], list_of_rights[interval_start:interval_end]))
                survive += list(survive_function(list_of_lefts[interval_start:interval_end],
                                    Z[interval_start:interval_end], list_of_rights[interval_start:interval_end]))

            #print("BITH: " + str(birth))
            birth = np.array(birth, dtype="bool")
            #print("BITH: " + str(birth))
            survive = np.array(survive, dtype="bool")
            current_input[...] = 0
            current_input[birth | survive] = 1
            #print(current_input)
            output.append(list(current_input))
        #output = np.array(output, dtype="uint8")
        #print(output)
        return output

class Rule:
    def __init__(self, number=0):
        """

        :param number: Corresponds to Wolfram number of elem. CA rules
        :return:
        """
        self.number = number

    def getRuleScheme(self):
        """
        :param rule_number:
        :return:
        """
        binrule = format(self.number, "08b")  # convert to binary, with fill of zeros until the string is of length 8

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

        scheme = self.getRuleScheme()
        output = scheme[(input_array[0], input_array[1], input_array[2])]

        return output

    def get_logical_expression(self):
        scheme = self.getRuleScheme()
        births = []
        survives = []
        for tup in scheme.keys():
            if tup[1] == 1 and scheme.get(tup) == 1:
                survives.append(tup)
            if tup[1] == 0 and scheme.get(tup) == 1:
                births.append(tup)

        print("Births: " + str(births))
        ## BIRTH
        birth_base_expressions = []
        for x, y, z in births:
            birth_base_expressions.append(
                "((left_list == "+str(x) + ") & (center_list == "+str(y) + ") & (right_list == "+str(z) + "))")
        lambda_expression = "("
        for i in range(len(birth_base_expressions)):
            lambda_expression += birth_base_expressions[i]
            if i!= len(birth_base_expressions)-1:
                lambda_expression += " | "
        lambda_expression += ")"
        print(lambda_expression)
        birth = lambda left_list, center_list, right_list: (eval(lambda_expression))

        ## SURVIVE
        survives_base_expressions = []
        for left, center, right in survives:
            survives_base_expressions.append(
                "((left_list == " + str(left) + ") & (center_list == " + str(center) + ") & (right_list == " + str(right) + "))")
        survives_expression = "("
        for i in range(len(survives_base_expressions)):
            survives_expression += survives_base_expressions[i]
            if i != len(survives_base_expressions) - 1:
                survives_expression += " | "
        survives_expression += ")"
        survives = lambda left_list, center_list, right_list: (eval(survives_expression))
        print(birth([0,0,1],
                    [0,0,0],
                    [1,0,0]))
        return birth, survives


