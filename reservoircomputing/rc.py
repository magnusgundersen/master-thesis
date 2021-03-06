"""
Module for reservoir computing.

The classifier and reservoir must implement the interfaces as described by the rc_interface.py-module
"""
import numpy as np
import random
import warnings

import timeit
import time

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
        The training_data must look like this:
        training_data = (X, Y)

        X =
        [
        x_1,
        x_2
        ]

        Y =
        [
        y_1,
        y_2
        ]



        :param training_data:
        :return:
        """

        input_X = training_data[0]

        output_Y = training_data[1]


        self.rc_helper.reset()
        rc_outputs = []  # One output for each time step

        for _input in input_X:
            rc_output = self.rc_helper.run_input(_input)
            rc_outputs.append(rc_output)
            # make training-set for the classifier:
            self.classifier_input_set.append(rc_output.classifier_output())

        self.classifier_output_set.extend(output_Y)
        return rc_outputs

    def train_classifier(self):
        self.classifier.fit(self.classifier_input_set, self.classifier_output_set)

    def predict(self, input_X):
        """

        The input consists of data:

        X =
        [
        x_1,
        x_2
        ]



        :param test_data:
        :return:
        """

        _outputs = []
        self.rc_helper.reset()
        for _input in input_X:  # input and output at each timestep
            rc_output = self.rc_helper.run_input(_input)
            classifier_input = rc_output.classifier_output()
            classifier_prediction = self.classifier.predict(np.array(classifier_input).reshape(1,-1))
            _outputs.append(classifier_prediction)

        return _outputs

    def predict_dynamic_sequence(self, input_data, pred_stop_signal):
        _outputs = []
        self.rc_helper.reset()
        current_input = 0
        input_size = len(input_data[0])
        #print(input_data[0][0])
        while True:  # input and output at each timestep
            try:
                rc_output = self.rc_helper.run_input(input_data[current_input])
            except:
                rc_output = self.rc_helper.run_input([0]*(input_size-1) + [1])  # If the input-size is done...
            classifier_input = rc_output.classifier_output()
            classifier_prediction = self.classifier.predict(np.array(classifier_input).reshape(1,-1))
            _outputs.append(classifier_prediction)
            if classifier_prediction == pred_stop_signal or current_input > 100: # Avoid inf. loop
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
        self.time_usage = 0
        self.time_counter = 0


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
        encoded_input = np.concatenate(encoded_inputs).ravel()

        # 4. step is to use transition to take previous steps into account
        if self.time_step > 0:  # No transition at first time step
            transitioned_data = self.time_transition.join(encoded_input, self.last_step_data, self.encoder)
        else:
            transitioned_data = encoded_input


        # 5. step is to propagate in CA reservoir
        before = time.time()
        all_propagated_data = self.reservoir.run_simulation(transitioned_data, self.I)
        #print("used: " + str(time.time() - before))
        self.time_usage += (time.time() - before)
        self.time_counter += 1
        #print("avg so far: " + str(self.time_usage/self.time_counter))
        previous_data = np.copy(self.last_step_data)
        self.last_step_data = all_propagated_data[-1]

        # 6. step is to create an output-object
        output = RCOutput()
        output.set_states(all_propagated_data, previous_data)

        self.time_step += 1

        return output















class RCOutput:
    """
    Class that contains the whole reservoir-computing process. May be used by the classifier for investigating.
    """

    def __init__(self):
        self.list_of_states = []
        self.transitioned_state = []
        self.flattened_states = []

    def set_states(self, all_states, transitioned_state):
        self.list_of_states = all_states
        self.transitioned_state = transitioned_state
        #self.flattened_states = np.concatenate(all_states[-4:-1]).ravel()
        self.flattened_states = np.concatenate(all_states)
        #self.flattened_states = all_states[1:-1]



    def classifier_output(self):
        #return self.flattened_states[::2]
        return self.flattened_states









