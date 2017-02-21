import random
import numpy as np
class RandomAdditionTimeTransition:
    def __init__(self):
        self.previous_inputs = {} # dict to contains the previous inputs to .....

    def join(self, _input, transition_input, encoder):
        return self.normalized_adding(transition_input, _input)

    def normalized_adding(self, transmission_input, _input):

        transmitted_output = []
        for i in range(len(transmission_input)):
            a = transmission_input[i]
            b = _input[i]

            if a == 1 and b == 1:
                transmitted_output.append(1)
            elif a == 1 and b == 0:  # Returning 1 gives good results
                transmitted_output.append(random.choice([0, 1])) #random.choice([0, 1])
            elif a == 0 and b == 1:
                transmitted_output.append(1)
            elif a == 0 and b == 0:
                transmitted_output.append(0)
        return transmitted_output

class RandomPermutationTransition:
    def __init__(self):
        pass

    def join(self, _input, _transition_input, encoder):

        size = len(_transition_input)


        mappings = encoder.mappings
        R = encoder.R

        adjusted_mappings = []
        for i in range(len(mappings)):
            new_mapping = []
            for integer in mappings[i]:
                new_mapping.append(integer+(i*(size//R))) # 8 ni
            adjusted_mappings.extend(new_mapping)

        new_input = ["x" for _ in range(size)]

        for index in adjusted_mappings:
            new_input[index] = _input[index]
        for i in range(len(new_input)):
            if new_input[i] == "x":
                new_input[i] = _transition_input[i]

        return new_input


class XORTimeTransition:
    def __init__(self):
        self.previous_inputs = {} # dict to contains the previous inputs to .....

    def join(self, _input, transition_input, encoder):
        return self.xor(transition_input, _input)

    def xor(self, transmission_input, _input):
        transmitted_output = np.bitwise_xor(transmission_input, _input)
        return transmitted_output

