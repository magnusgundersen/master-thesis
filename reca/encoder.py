from reservoircomputing import rc_interface as rcif
import random


class RandomMappingEncoder(rcif.RCEncoder):
    def __init__(self, parallelizer):
        super().__init__()
        self.R = 1
        self.C = 1
        self.P = parallelizer.P
        self.parallel_size_policy = parallelizer.parallel_size_policy
        self.encoding_scheme = "separate"

    def create_mappings(self, input_length):
        """
        Facilitates having a fixed mapping
        :param input_length:
        :return:
        """
        #print("Creating mappings!")
        list_of_mappings = []
        self.input_length = input_length
        num_list = [x for x in range(input_length*self.C)]
        num_list2 = num_list[:]

        for _ in range(self.R):
            random.shuffle(num_list2)
            list_of_mappings.append(num_list2[:self.input_length])

        # EXPERIMENTAL: PADDING
        """
        for i in range(len(list_of_mappings)):
            new_list = []
            for j in range(len(list_of_mappings[i])):
                new_list.extend([list_of_mappings[i][j],0])
            list_of_mappings[i] = new_list
            if i ==0:
                print(new_list)
        """
        self.mappings = list_of_mappings


    def encode_input(self, _input):
        """

        :param _input:
        :return:
        """
        encoded_input = []
        if len(_input) != self.input_length:
            raise ValueError("Wrong input-length to encoder!")

        for i in range(self.R):
            temp_enc_list = [0 for _ in range(len(_input)*self.C)]
            for j in range(len(_input)):
                temp_enc_list[self.mappings[i][j]] = _input[j]

            encoded_input.append(temp_enc_list)
        #print(self.P)

        if self.parallel_size_policy == "unbounded":
            for _ in range(self.P-1):
                encoded_input += encoded_input
        return encoded_input


    def encode_output(self, _output):
        # Flatten
        _output = [ca_val for sublist in _output for ca_val in sublist]
        return _output

class ParallelNonUniformEncoder:  # Consider renaming/rebranding to a "rule-governer"
    def __init__(self, ca_rules, parallel_size_policy="unbounded"):
        self.P = len(ca_rules)
        self.ca_rules = ca_rules
        self.parallel_size_policy = parallel_size_policy
        self.rule_dict = {}



    def encode(self, _input):
        new_input = []
        rule_dict = {}

        if self.parallel_size_policy == "unbounded":
            size = len(_input)/self.P
            new_input = _input
            for i in range(self.P):
                rule_dict[i*size, (i+1)*(size-1)] = self.ca_rules[i]


        elif self.parallel_size_policy == "bounded":
            new_input = _input
            size = len(_input)
            pieces = size // self.P  # devide in even parts
            for i in range(self.P):
                rule_dict[(i*pieces, ((i+1)*pieces)-1)] = self.ca_rules[i]
                # TODO: MAJOR PROBLEM if its an odd number of cells


        return new_input, rule_dict



