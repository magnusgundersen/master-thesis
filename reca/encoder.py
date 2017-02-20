from reservoircomputing import rc_interface as rcif
import random


class RandomMappingEncoder(rcif.RCEncoder):
    def __init__(self):
        super().__init__()
        self.R = 1
        self.C = 1
        self.encoding_scheme = "separate"  # Whether to keep the reservoirs separate or not

        self.unencoded_input_length = 0

    def create_mappings(self, input_length):
        """
        Facilitates having a fixed mapping

        If the parameter C is larger than 1, the mapping vector will be smaller than the size of the input-vector.

        :param input_length:
        :return:
        """
        list_of_mappings = []
        self.unencoded_input_length = input_length
        num_list = [x for x in range(self.unencoded_input_length*self.C)]  # list of number to be shuffled
        num_list2 = num_list[:]  # copy list to have a ref to the original

        for _ in range(self.R):
            random.shuffle(num_list2)  # New random mapping for each R.
            list_of_mappings.append(
                num_list2[:self.unencoded_input_length])  # Only to the size of the input, and let there be C padding


        self.mappings = list_of_mappings


    def encode_input(self, _input):
        """

        :param _input:
        :return:
        """
        encoded_input = []
        if len(_input) != self.unencoded_input_length:
            raise ValueError("Wrong input-length to encoder!")

        for i in range(self.R):
            temp_enc_list = [0 for _ in range(len(_input)*self.C)]
            for j in range(len(_input)):
                temp_enc_list[self.mappings[i][j]] = _input[j]

            encoded_input.append(temp_enc_list)
        #print(self.P)

        #if self.parallel_size_policy == "unbounded":
        #    for _ in range(self.P-1):
        #        encoded_input += encoded_input
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



