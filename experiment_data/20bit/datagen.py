import random
import itertools
from itertools import combinations
def generate_20_bit_file(distractor_period, training_set_size, testing_set_size):

    content = ""

    input_signals = ["a_1", "a_2", "a_3", "a_4", "a_5"]
    patterns = set()


    while len(patterns) < (training_set_size + testing_set_size):
        pattern = []
        for j in range(10):
            pattern.append(random.choice(input_signals))
        patterns.add(tuple(pattern))

    patterns = list(patterns)
    training_set = patterns[:training_set_size]
    testing_set = patterns[training_set_size:]

    for train_ex in training_set:
        text_content_ex = ""
        for signal in train_ex:
            if signal == "a_1":
                text_content_ex += "10000"
            elif signal == "a_2":
                text_content_ex += "01000"
            elif signal == "a_3":
                text_content_ex += "00100"
            elif signal == "a_4":
                text_content_ex += "00010"
            elif signal == "a_5":
                text_content_ex += "00001"




    """
    for i in range(training_set_size + testing_set_size):
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
    """
    #with open(str(n)+ "_bit_" + str(distractor_period) + "_dist_" + str(number_of_training_sets),'w+') as f:
    #    f.write(content)

def get_input_by_a1_value(a1_value):
    if a1_value == 1:
        return "1000", "100"
    else:
        return "0100", "010"


if __name__ == "__main__":
    generate_20_bit_file(100, 20000, 1000)


