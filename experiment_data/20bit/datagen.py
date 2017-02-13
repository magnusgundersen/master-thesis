import random
import itertools
from itertools import combinations
def generate_20_bit_file(distractor_period, number_of_training_sets):

    content = ""

    input_signals = ["a_1", "a_2", "a_3", "a_4", "a_5"]




    for i in range(number_of_training_sets):
        training_set = ""
        a1_value = 1
        signal = []

        # n bit signal
        for j in range(n):
            _input, _corresponding_output = get_input_by_a1_value(a1_value[j])
            _output = "001"
            training_set += _input + " " + _output + "\n"
            signal.append(_corresponding_output)

        # Distractor period
        for j in range(distractor_period-1):
            _input = "0010"
            _output = "001"
            training_set += _input + " " + _output + "\n"

        # Cue signal
        _input = "0001"
        _output = "001"
        training_set += _input + " " + _output + "\n"

        # repeated n bit signal
        for signal_entry in signal:
            _input = "0010"
            _output = signal_entry
            training_set += _input + " " + _output + "\n"


        content += training_set + "\n"

    with open(str(n)+ "_bit_" + str(distractor_period) + "_dist_" + str(number_of_training_sets),'w+') as f:
        f.write(content)

def get_input_by_a1_value(a1_value):
    if a1_value == 1:
        return "1000", "100"
    else:
        return "0100", "010"
generate_n_bit_file(5, 200, 32)


