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

    training_text = ""
    testing_text = ""

    for ex in training_set:
        training_text += create_text_file_data(ex, distractor_period) + "\n"

    for ex in testing_set:
        testing_text += create_text_file_data(ex, distractor_period) + "\n"




    with open("20_bit_train_" + str(distractor_period) + "_dist_" + str(training_set_size),'w+') as f:
        f.write(training_text)
    with open("20_bit_test_" + str(distractor_period) + "_dist_" + str(testing_set_size),'w+') as f:
        f.write(testing_text)

def create_text_file_data(ex, distractor_period):
    text_content_ex = ""
    output_signals = []
    output_wait_signal = "000001"

    input_distractor_signal = "0000010"

    cue_signal = "0000001"

    for signal in ex:
        _input, _output = get_input_by_a_value(signal)
        output_signals.append(_output)
        text_content_ex += _input + " " + output_wait_signal + "\n"

    for period in range(distractor_period-1):  # -1 because Jaeger says so
        text_content_ex += input_distractor_signal + " " + output_wait_signal + "\n"

    text_content_ex += cue_signal + " " + output_wait_signal + "\n"

    for signal in output_signals:
        text_content_ex += input_distractor_signal + " " + signal + "\n"

    return text_content_ex

def get_input_by_a_value(a):
    if a == "a_1":
        return "1000000", "100000"
    if a == "a_2":
        return "0100000", "010000"
    if a == "a_3":
        return "0010000", "001000"
    if a == "a_4":
        return "0001000", "000100"
    if a == "a_5":
        return "0000100", "000010"



if __name__ == "__main__":
    generate_20_bit_file(10, 120, 20)


