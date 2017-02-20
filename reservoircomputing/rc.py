"""
Module for reservoir computing.

The classifier and reservoir must implement the interfaces as described by the rc_interface.py-module
"""
import numpy as np
import random
import warnings




class ReservoirComputingFramework:
    """
    Class used to execute reservoir computing
    It is responsible for

    The reservoir must implement the reservoir-interface
    The classifier must implement the classifier-interface


    """
    def __init__(self):

        self.rc_helper = None

        self.classifier_input_set = []
        self.classifier_output_set = []

    def set_helper(self, rc_helper):
        self.rc_helper = rc_helper
        self.classifier = rc_helper.config.classifier



    def run_example_simulation(self, _input, iterations):
        # TODO: Solve differently

        encoded_input = self.encoder.encode_input(_input)
        unencoded_output = []
        for _input in encoded_input:
            reservoir_output = self.reservoir.run_simulation(_input, iterations)
            #reservoir_output = [ca_val for sublist in reservoir_output for ca_val in sublist]  # flatten
            unencoded_output.append(reservoir_output)

        #encoded_output = self.encoder.encode_output(unencoded_output)
        return unencoded_output



    def fit_to_data(self, training_data):
        """
        Fit the RC-system to the RC-problem using the RC-helper
        The input consists of data:

        temporal_data =
        [
        (input, output),
        (input, output)
        ]

        or:

        non_temporal_data =
        [
        (input, output)
        ]

        :param training_data:
        :return:
        """

        self.rc_helper.reset()
        rc_outputs = []  # One output for each time step
        for _input, _output in training_data: # input and output at each timestep
            rc_output = self.rc_helper.run_input(_input)
            rc_outputs.append(rc_output)
            # make training-set for the classifier:
            self.classifier_input_set.append(rc_output.classifier_output())
            self.classifier_output_set.append(_output)

        return rc_outputs




    def train_classifier(self):
        self.classifier.fit(self.classifier_input_set, self.classifier_output_set)

    def predict(self, test_data):
        """

                The input consists of data:

                temporal_data =
                [
                (input, output),
                (input, output)
                ]

                or:

                non_temporal_data =
                [
                (input, output)
                ]

                :param training_data:
                :return:
        """

        _outputs = []
        classifier_input_set = []
        self.rc_helper.reset()
        for _input, _ in test_data:  # input and output at each timestep
            rc_output = self.rc_helper.run_input(_input)
            classifier_input = rc_output.flattened_states
            classifier_prediction = self.classifier.predict(classifier_input)
            _outputs.append(classifier_prediction)

        return _outputs

    def predict_dynamic_sequence(self, input_data, pred_stop_signal):
        _outputs = []
        classifier_input_set = []
        self.rc_helper.reset()
        current_input = 0
        input_size = len(input_data[0][0])
        print(input_data[0][0])
        while True:  # input and output at each timestep
            try:
                rc_output = self.rc_helper.run_input(input_data[current_input][0])
            except:
                rc_output = self.rc_helper.run_input([0]*input_size)
            classifier_input = rc_output.flattened_states
            classifier_prediction = self.classifier.predict(np.array(classifier_input).reshape(1,-1))
            _outputs.append(classifier_prediction)
            if classifier_prediction == "000000000000000000000000000000000000000000000000000000001" or current_input > 1000: # Avoid inf. loop
                break

            current_input += 1
        return _outputs





class RCHelper:
    """
    Helper class to manage the execution of the RC-cycle.
    """
    def __init__(self, external_config):
        self.config = external_config  # Config (i.e. reca-config)
        self.encoder = self.config.encoder  # Encoder that processes the input
        self.time_transition = self.config.time_transition  # How the previous inputs are echoed through the reservoir
        self.reservoir = self.config.reservoir
        self.I = self.config.I # Number of iterations (CA)
        self.parallelizer = self.config.parallelizer # CA parallelizer


    def reset(self):
        self.time_step = 0
        self.previous_data = []
        self.last_step_data = []

    def run_input(self, _input):  # Partly on
        # Run input that is deptandant on previous inputs


        # 1. step is to consider if the reservoir landscape is parallelized
        # TODO: currently not implemented
        #print("INPUT: " + str(_input))
        # 2. Step is to encode the input
        encoded_inputs = self.encoder.encode_input(_input)  # List of lists
        # 3. Step is to concat or keep the inputs by themselves
        # TODO: Remove this if you want to be able to have separate reservoirs!
        encoded_input = [val for sublist in encoded_inputs for val in sublist]
        pre_enc = encoded_input[:]
        encoded_input, rule_dict = self.parallelizer.encode(encoded_input)

        # 4. step is to use transition to take previous steps into account
        if self.time_step > 0:  # No transition at first time step
            transitioned_data = self.time_transition.join(encoded_input, self.last_step_data, self.encoder)
        else:
            transitioned_data = encoded_input

          # ajour

        # 5. step is to propagate in CA reservoir
        all_propagated_data = self.reservoir.run_simulation(transitioned_data, self.I)
        previous_data = self.last_step_data[:]
        self.last_step_data = all_propagated_data[-1]

        # 6. step is to create an output-object
        output = RCOutput()
        output.set_states(all_propagated_data, previous_data)

        self.time_step += 1

        return output

    def training_finished(self):
        pass
















class RCOutput:
    """
    Class that contains the whole reservoir-computing process. May be used by the classifier for investigating.

    TODO: Facilitate that the RCOutput may be "sparse" (Bio-inspired)
    """

    def __init__(self):
        self.list_of_states = []
        self.transitioned_state = []
        self.flattened_states = []

    def set_states(self, all_states, transitioned_state):
        self.list_of_states = all_states
        self.transitioned_state = transitioned_state
        self.flattened_states = [state_val for sublist in all_states for state_val in sublist]


    def classifier_output(self):
        return self.flattened_states









