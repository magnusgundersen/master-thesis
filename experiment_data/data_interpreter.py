# Creates a ReCAProblem from the dataset.
import pickle
import numpy as np
import os
import random
import pprint
import xml.etree.ElementTree as ET
import scipy.sparse as sprs
import scipy.io as sci_io
#from speech import Speech, SpeakerStatistics, SpeakerStatisticsCollection
from PIL import Image
#import Image
file_location = os.path.dirname(os.path.realpath(__file__))

class TranslationBuilder:
    def __init__(self):
        """
        utf-8
        """
        self.training_set_fraction = 0.95
        self.testing_set_fraction = 1-self.training_set_fraction

        self.prod_end_signal = ""

        self.english_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                            "s", "t", "u", "v", "w", "x", "y", "z"]

        self.german_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                           "s", "t", "u", "v", "w", "x", "y", "z", "ä", "ö", "ü"]

        self.french_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                           "s", "t", "u", "v", "w", "x", "y", "z"]

        self.common_signs = [" ", ".", ",", ":", ";", "-", "!", "?", "'", "(", ")", "%", "/", '"', "'", "+"]
        self.arabic_numbers = [str(i) for i in range(10)]

        self.english_allowed = self.english_alphabet + self.common_signs + self.arabic_numbers
        self.german_allowed = self.german_alphabet + self.common_signs + self.arabic_numbers

        self.language = "german"  #ONLY ONE IMPLEMENTED

    def get_training_data(self):

        data_set = []
        eng_lines, ger_lines = (self.read_translation_files("de"))
        short_ger_lines = []
        short_eng_lines = []
        for i in range(len(ger_lines)):
            ger_line = ger_lines[i]
            eng_line = eng_lines[i]
            if 20 < len(ger_line) < 25:
                short_ger_lines.append(ger_line)
                short_eng_lines.append(eng_line)

                if len(short_ger_lines) == 20000: break

        for i in range(len(short_eng_lines)):
            _inputs, _outputs = self.create_bin_data(short_eng_lines[i], self.english_allowed, short_ger_lines[i], self.german_allowed)
            string_outputs = []
            for _output in _outputs:
                string_result = ""
                for char in _output:
                    string_result += str(char)
                string_outputs.append(string_result)
            #_inputs = self.binarize(_inputs)
            data_set.append((_inputs, np.array(string_outputs)))

        return data_set[:int(self.training_set_fraction*len(data_set))]

    def get_testing_data(self):

        data_set = []
        eng_lines, ger_lines = (self.read_translation_files("de"))
        short_ger_lines = []
        short_eng_lines = []
        for i in range(len(ger_lines)):
            ger_line = ger_lines[i]
            eng_line = eng_lines[i]
            if 20 < len(ger_line) < 25:
                short_ger_lines.append(ger_line)
                short_eng_lines.append(eng_line)

                if len(short_ger_lines) == 20000: break

        for i in range(len(short_eng_lines)):
            _inputs, _outputs = self.create_bin_data(short_eng_lines[i], self.english_allowed, short_ger_lines[i],
                                                  self.german_allowed)
            string_outputs = []
            for _output in _outputs:
                string_result = ""
                for char in _output:
                    string_result += str(char)
                string_outputs.append(string_result)
            #_inputs = self.binarize(_inputs)

            data_set.append((_inputs, np.array(string_outputs)))

        return data_set[int(self.training_set_fraction*len(data_set)):]

    def get_training_data_old(self):

        if self.language == "german":
            dataset = []  #
            batches_location = file_location + "/translation/en-de/"
            lang_locations = batches_location + "/de"
            eng_locations = batches_location + "/en"
            lang_files = os.listdir(lang_locations)
            eng_files = os.listdir(eng_locations)

            #print(lang_files)
            number_of_examples = len(lang_files)
            number_of_examples = 10

            for i in range(number_of_examples):

                _inputs = sci_io.mmread(lang_locations + "/" + lang_files[i]).toarray()

                _outputs = sci_io.mmread(eng_locations + "/" + eng_files[i]).toarray()
                string_outputs = []
                for _output in _outputs:
                    string_result = ""
                    for char in _output:
                        string_result+= str(char)
                    string_outputs.append(string_result)
                dataset.append((_inputs, np.array(string_outputs)))


            print("Finished retreiving training data")
            return dataset
        else:
            raise ValueError("Language not implemented: " + str(self.language))

    def get_testing_data_old(self):
        print("getting testing data")
        file_location = os.path.dirname(os.path.realpath(__file__))
        if self.language == "german":
            dataset = []  #
            batches_location = file_location + "/translation/en-de/"
            lang_locations = batches_location + "/de"
            eng_locations = batches_location + "/en"
            lang_files = os.listdir(lang_locations)
            eng_files = os.listdir(eng_locations)

            number_of_examples = len(lang_files)
            number_of_examples = 10

            for i in range(number_of_examples):

                _inputs = sci_io.mmread(lang_locations + "/" + lang_files[19000-i]).toarray()


                _outputs = sci_io.mmread(eng_locations + "/" + eng_files[19000-i]).toarray()


                string_outputs = []
                for _output in _outputs:
                    string_result = ""
                    for char in _output:
                        string_result += str(char)
                    string_outputs.append(string_result)

                dataset.append((_inputs, np.array(string_outputs)))





            print("Finished retreiving testing data")
            return dataset
        else:
            raise ValueError("Language not implemented: " + str(self.language))

    def get_pred_end_signal(self):
        if self.language == "german":
            return "000000000000000000000000000000000000000000000000000000001"  # TODO: make dynamic

    def convert_from_bit_sequence_to_string(self, bit_sequence, language):
        sentence = ""
        if len(bit_sequence)<20:
            bit_sequence = self.unbinarize(bit_sequence)
        if language == "german":

            for bit_string in bit_sequence:
                alphabet_index = 0
                for _index in bit_string:
                    if _index == '1':  # Correct char
                        try:
                            sentence += self.german_allowed[alphabet_index]
                        except:
                            pass  # Wait or finish signal given
                        break  # Found correct character
                    else:
                        alphabet_index += 1

        elif language == "english":
            for bit_string in bit_sequence:
                alphabet_index = 0
                for _index in bit_string:
                    if _index == '1':  # Correct char
                        try:
                            sentence += self.english_allowed[alphabet_index]
                        except:
                            pass  # Wait or finish signal given
                        break  # Found correct character
                    else:
                        alphabet_index += 1
        return sentence

    def read_translation_files(self, language, breakoff=False):
        file_location = os.path.dirname(os.path.realpath(__file__))
        usable_eng_lines = []
        usable_lang_lines = []

        with open(file_location+"/translation/txt/" + "europarl-v7."+language+"-en."+language, "r", encoding='utf8') as lang,\
                open(file_location+"/translation/txt/" + "europarl-v7." + language + "-en." + "en", "r", encoding='utf8') as eng:
            lang = lang.readlines()
            eng = eng.readlines()

            if len(lang) != len(eng):
                raise ValueError("Both languages must have the same number of lines!")

            for i in range(len(lang)):

                eng_line = eng[i]
                lang_line = lang[i]
                if (eng_line == "\n" or lang_line == "\n") or (eng_line[0] == "." or lang_line[0] == "."):
                    continue
                usable_eng_lines.append(eng_line)
                usable_lang_lines.append(lang_line)

                if breakoff and i == 20000:
                    break




        return (usable_eng_lines, usable_lang_lines)

    def generate_translation_data(self):
        ger_lines, eng_lines = (self.read_translation_files("de"))
        print(len(ger_lines))
        print(len(eng_lines))



        translation_data = []
        for i in range(len(ger_lines)):
            eng, ger = (self.create_bin_data(eng_lines[i], self.english_allowed, ger_lines[i], self.german_allowed))
            #translation_data.append(example)
            eng = sprs.csr_matrix(eng, dtype="int8")
            ger = sprs.csr_matrix(ger, dtype="int8")
            #np.save("translation/en-de/en/" + str(i) + ".en", eng)
            #np.save("translation/en-de/de/" + str(i) + ".de", ger)
            sci_io.mmwrite("translation/en-de/en/" + str(i) + ".en", eng)
            sci_io.mmwrite("translation/en-de/de/" + str(i) + ".de", ger)



        #print(translation_data)
        #translation_data = np.array(translation_data)
        #print(translation_data.shape)
        #pickle.dump(translation_data, open('save.p', 'wb'))

    def create_bin_data(self, source_sentence, source_alphabet, target_sentence, target_alphabet):
        #print("source sentence; " + source_sentence)
        #print("english allowed: " + str(self.english_allowed))
        #print("target sentence; " + target_sentence)
        sequence_length = len(source_sentence) + len(target_sentence) + 1  # : +1: prediction end signal
        source_array = np.zeros((sequence_length, len(source_alphabet)+1), dtype="uint8")  # + 1: end of source sentence
        target_array = np.zeros((sequence_length, len(target_alphabet)+1+1), dtype="uint8")  # + 1 +1 : wait and end of sentence

        # Create source and target array inputs/outputs for source sentence
        for i in range(len(source_sentence)):
            for j in range(len(source_alphabet)):
                character = source_alphabet[j]
                if source_sentence[i].lower() == character.lower():
                    source_array[i][j] = 1
                    break
            target_array[i][-2] = 1  # wait signal for each input-timestep

        target_sentence_pointer = 0
        for i in range(len(source_sentence), len(source_sentence) + len(target_sentence)):
            source_array[i][-1] = 1  # source sentence ended, produce translated sentence signal
            for j in range(len(target_alphabet)):
                character = target_sentence[target_sentence_pointer]
                if target_alphabet[j] == character.lower():
                    target_array[i][j] = 1
                    break
            target_sentence_pointer += 1


        ## prediction end signal:
        source_array[-1][-1] = 1
        target_array[-1][-1] = 1
        signal = np.zeros(len(target_array[-1]))
        signal[-1] = 1
        self.prediction_end_signal = signal # End of sentence

        #print(source_array.shape)
        #print(target_array.shape)

        return source_array, target_array

    def binarize(self, _inputs):
        return_array = []
        for time_step in _inputs:
            # Find the "1":
            index = 0
            for _input in time_step:
                if _input == 1:
                    break
                index += 1
            binary_number = bin(index)[2:]  #Rm 0b
            binary_number = binary_number.zfill(7) # pad to 64 bit number
            return_array.append([int(bin_char) for bin_char in binary_number])
        return return_array

    def unbinarize(self, _inputs):
        return_array = []
        for time_step in _inputs:
            #print(time_step)
            index = int("".join([str(char) for char in time_step]), 2)
            #print(index)
            new_time_step = np.zeros(53)
            new_time_step[index] = 1
            return_array.append(new_time_step)

        return return_array


class CIFARBuilder:
    def get_cifar_data(self):
        batch1 = {}
        with open("cifar10/data_batch_1",'rb') as f:
            batch1 = pickle.load(f, encoding='bytes')
        print(batch1)
        #data = batch1.get(b'data')[5].reshape(3,32,32).transpose(1,2,0)
        data = batch1.get(b'data')
        labels = batch1.get(b'labels')
        #img = Image.fromarray(data, 'RGB')
        #img.save('my.png')
        #img.show()

        data_lenght = len(data)
        data_lenght = 100
        data_string = ""

        for i in range(data_lenght):
            data_string += self.create_bin_data(data[i], labels[i], None)
            data_string += "\n\n"

        with open("cifar.data", "w+") as f:
            f.write(data_string)

    def create_bin_data(self, img_array, img_class, weight_matrix):
        # C-style flattening
        bin_string = ""
        for chn in img_array:
            if chn > 100:  # THRESHOLD
                bin_string += "1"
            else:
                bin_string += "0"

        bin_string += " "
        for j in range(10):
            if j == img_class:
                bin_string += "1"
            else:
                bin_string += "0"

        return bin_string


class FiveBitBuilder:
    def __init__(self, dist_period=200, testing_ex=32, training_ex=32):
        self.dist_period = dist_period
        self.no_training_ex = training_ex
        self.no_testing_ex = testing_ex

    def get_training_data(self):
        dataset = []
        with open(file_location+"/5bit/5_bit_" + str(self.dist_period) + "_dist_32", "r") as f:
            content = f.readlines()
            training_inputs = []
            training_ouputs = []
            for line in content:
                if line == "\n":
                    dataset.append((np.array(training_inputs, dtype="uint8"), np.array(training_ouputs)))
                    training_inputs = []
                    training_ouputs = []
                else:
                    _input, _output = line.split(" ")
                    training_inputs.append([int(number) for number in _input])
                    training_ouputs.append(_output[0:-1])  # class is text
        dataset = dataset[:self.no_training_ex]
        return dataset


    def get_testing_data(self):
        dataset = []
        with open(file_location+"/5bit/5_bit_" + str(self.dist_period) + "_dist_32", "r") as f:
            content = f.readlines()
            training_inputs = []
            training_ouputs = []
            for line in content:
                if line == "\n":
                    dataset.append((np.array(training_inputs, dtype="uint8"), np.array(training_ouputs)))
                    training_inputs = []
                    training_ouputs = []
                else:
                    _input, _output = line.split(" ")
                    training_inputs.append([int(number) for number in _input])
                    training_ouputs.append(_output[0:-1])  # class is text

        dataset = dataset[(32-self.no_testing_ex):]
        return dataset


class TwentyBitBuilder:
    def __init__(self, distractor_period, training_ex, testing_ex):
        self.dist_period = distractor_period
        self.no_training_ex = training_ex
        self.no_testing_ex = testing_ex

    def get_training_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))
        dataset = []  # list of numpy-arrays

        with open(file_location+"/20bit/20_bit_train_" + str(self.dist_period) + "_dist_" + str(self.no_training_ex), "r") as f:
            content = f.readlines()
            training_inputs = []
            training_ouputs = []
            for line in content:
                if line == "\n":
                    dataset.append((np.array(training_inputs, dtype="uint8"), np.array(training_ouputs)))
                    training_inputs = []
                    training_ouputs = []
                else:
                    _input, _output = line.split(" ")
                    training_inputs.append([int(number) for number in _input])
                    training_ouputs.append(_output[0:-1])  # class is text

        return dataset

    def get_testing_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

        dataset = []
        with open(file_location+"/20bit/20_bit_test_" + str(self.dist_period) + "_dist_"+str(self.no_testing_ex), "r") as f:
            content = f.readlines()
            training_inputs = []
            training_ouputs = []
            for line in content:
                if line == "\n":
                    dataset.append((np.array(training_inputs, dtype="uint8"), np.array(training_ouputs)))
                    training_inputs = []
                    training_ouputs = []
                else:
                    _input, _output = line.split(" ")
                    training_inputs.append([int(number) for number in _input])
                    training_ouputs.append(_output[0:-1])  # class is text
        return dataset


class JapaneseVowelsBuilder:
    def __init__(self):
        pass

    def read_test_and_train_files(self):
        training_set = []
        testing_set = []

        with open(file_location +"/japanese_vowels/size_ae.train", "r") as f:
            sizes = f.readlines()[0]
            sizes = sizes.split(" ")
            sizes[-1] = sizes[-1][:-1]  #RM newline
            sizes = [int(size) for size in sizes]

        with open(file_location +"/japanese_vowels/ae.train", "r") as f:
            lines = f.readlines()
            training_set = self._get_speaker_utterances_from_size_and_lines(lines, sizes)


        with open(file_location +"/japanese_vowels/size_ae.test", "r") as f:
            sizes = f.readlines()[0]
            sizes = sizes.split(" ")
            sizes[-1] = sizes[-1][:-1]  #RM newline
            sizes = [int(size) for size in sizes]


        with open(file_location +"/japanese_vowels/ae.test", "r") as f:
            lines = f.readlines()
            testing_set = self._get_speaker_utterances_from_size_and_lines(lines, sizes)

        return training_set, testing_set

    def _get_speaker_utterances_from_size_and_lines(self, lines, sizes):
        """
        Extracts the utterances from the nine speakers, each of the given size
        :param lines:
        :param sizes:
        :return:
        """
        #print(lines)
        all_lines = []

        for line in lines:
            line = line.split(" ")
            all_lines.append(line)

        all_utterances = []
        temp_utterance = []
        for line in all_lines:
            #print(line)
            if line[0] == "\n":  # newline between each utterance
                all_utterances.append(temp_utterance)
                temp_utterance = []
            else:
                temp_utterance.append(line)

        #print(len(all_utterances))
        current_point = 0
        speaker_lines = []
        for i in range(len(sizes)):

            speaker_lines.append(all_utterances[current_point:current_point+sizes[i]])
            current_point += sizes[i]

        #print(len(speaker_lines[0]))
        return speaker_lines

    def _create_binary_data(self, utterance, speaker_number):
        """
        Uses the binarization scheme to create a training-set example
        :param utterance: A list of numpy-arrays of size 12, and data type floats.
        :return:
        """
        _input = []
        number_of_speakers = 9
        _output = []
        resolution = 2
        eos_signal = [0]*12*resolution + [1]
        wait_signal = [0]*number_of_speakers + [1]
        label_signal = [0]*(speaker_number) + [1] + [0]*((number_of_speakers-1)-speaker_number) + [0]



        for time_step in utterance:

            binary_version = [self._binarize(float_number, resolution) for float_number in time_step[:-1]] + [[0]] # Not EOS
            binary_version = list(np.concatenate(binary_version))
            _input.append(binary_version)
            _output.append(wait_signal)

        _input.append(eos_signal)
        _output.append(label_signal)
        #print(_input)
        #_input = np.array(_input, dtype="uint8")
        #_output = np.array(_output, dtype="uint8")
        return _input, _output

    def _binarize(self, input_float, resolution):
        try:
            input_float = float(input_float)
        except:
            raise ValueError("error on binarzation")

        if resolution==2:
            if input_float<-1:
                return [0,0]
            elif input_float<0:
                return [0,1]
            elif input_float<1:
                return [1,0]
            else:
                return [1, 1]

        elif resolution==3:
            if input_float<-2:
                return [0,0,0]
            elif input_float<-1.5:
                return [0,0,1]
            elif input_float<-1:
                return [0,1,0]
            elif input_float<-0.5:
                return [0,1,1]
            elif input_float<0.5:
                return [1,0,0]
            elif input_float<1.0:
                return [1,0,1]
            elif input_float<1.5:
                return [1,1,0]
            else:
                return [1, 1, 1]
        #return [random.choice([0, 1]) for _ in range(resolution)]

    def get_training_data(self):
        training_data, _ = self.read_test_and_train_files()
        dataset = []  #

        for i in range(len(training_data)):
            _inputs = []
            _outputs = []
            for utterance in training_data[i]:
                _inputs, _outputs = self._create_binary_data(utterance, i)

                string_outputs = []
                for _output in _outputs:
                    string_result = ""
                    for char in _output:
                        string_result+= str(char)
                    string_outputs.append(string_result)
                dataset.append((_inputs, np.array(string_outputs)))

        #print(len(dataset))

        return dataset

    def get_testing_data(self):
        _, testing_data = self.read_test_and_train_files()
        dataset = []  #


        for i in range(len(testing_data)):
            _inputs = []
            _outputs = []
            for utterance in testing_data[i]:
                _inputs, _outputs = self._create_binary_data(utterance, i)

                string_outputs = []
                for _output in _outputs:
                    string_result = ""
                    for char in _output:
                        string_result+= str(char)
                    string_outputs.append(string_result)
                dataset.append((_inputs, np.array(string_outputs)))


        return dataset

class FiveBitAndDensityBuilder:
    """
    Solve both five bit memory task, and the density task at the same time.
    Idea of the task is to emulate a complex problem (i.e. translation)
    """
    def __init__(self, dist_period=10, testing_ex=32, training_ex=32):
        self.dist_period = dist_period
        self.no_training_ex = training_ex
        self.no_testing_ex = testing_ex

        self.density_signals = 5  # Must be an odd number


    def generate_density_dataset(self):
        """

        :return:
        """
        input_length = 5
        cue_length = 1
        output_length = 5
        data_set = []

        for i in range(self.no_training_ex + self.no_testing_ex):
            example_input = []
            example_output = []
            density_count = []
            for j in range(input_length):
                _input = np.random.randint(2, size=self.density_signals)
                density_count.extend(list(_input))
                _output = "00"
                example_input.append(_input)
                example_output.append(_output)

            ones = density_count.count(1)
            zeros = density_count.count(0)

            for j in range(self.dist_period):
                _input = np.zeros(self.density_signals, dtype="uint8")
                _output = "00"
                example_input.append(_input)
                example_output.append(_output)

            for j in range(cue_length+3):
                _input = np.zeros(self.density_signals, dtype="uint8")
                _output = "00"
                example_input.append(_input)
                example_output.append(_output)

            for j in range(output_length-4):
                _input = np.zeros(self.density_signals, dtype="uint8")
                if zeros > ones:
                    _output = "10"
                else:
                    _output = "01"
                example_input.append(_input)
                example_output.append(_output)
            data_set.append((np.array(example_input, dtype="uint8"), np.array(example_output)))
        return data_set



    def get_training_data(self):
        five_bit_and_density_data = []
        five_bit_data = self.get_five_bit_data()
        # Hacked:
        while len(five_bit_data) < self.no_training_ex:
            five_bit_data.extend(five_bit_data[:])
        density_data = self.generate_density_dataset()

        for i in range(self.no_training_ex):
            _inputs, _outputs = five_bit_data[i]
            new_inputs = []
            new_outputs = []
            density_data_inputs, density_data_outputs = density_data[i]

            for j in range(len(_inputs)):
                new_inputs.append(np.append(_inputs[j], density_data_inputs[j]))
                new_outputs.append(np.array(str(_outputs[j])+ str(density_data_outputs[j])))

            five_bit_and_density_data.append((np.array(new_inputs), np.array(new_outputs)))

        return five_bit_and_density_data
    def get_five_bit_data(self):
        dataset = []
        with open(file_location+"/5bit/5_bit_" + str(self.dist_period) + "_dist_32", "r") as f:
            content = f.readlines()
            training_inputs = []
            training_ouputs = []
            for line in content:
                if line == "\n":
                    dataset.append((np.array(training_inputs, dtype="uint8"), np.array(training_ouputs)))
                    training_inputs = []
                    training_ouputs = []
                else:
                    _input, _output = line.split(" ")
                    training_inputs.append([int(number) for number in _input])
                    training_ouputs.append(_output[0:-1])  # class is text
        return dataset


    def get_testing_data(self):
        five_bit_and_density_data = []
        five_bit_data = self.get_five_bit_data()
        # Hacked:
        while len(five_bit_data) < self.no_testing_ex:
            five_bit_data.extend(five_bit_data[:])
        density_data = self.generate_density_dataset()

        for i in range(self.no_testing_ex):
            _inputs, _outputs = five_bit_data[i]
            new_inputs = []
            new_outputs = []
            density_data_inputs, density_data_outputs = density_data[i]

            for j in range(len(_inputs)):
                new_inputs.append(np.append(_inputs[j], density_data_inputs[j]))
                new_outputs.append(np.array(str(_outputs[j]) + str(density_data_outputs[j])))

            five_bit_and_density_data.append((np.array(new_inputs), np.array(new_outputs)))

        return five_bit_and_density_data


if __name__ == "__main__":
    #translator = TranslationBuilder()
    #translator.get_training_data()
    #translator.create_efficient_data_to_file()
    #translator.generate_translation_data()
    #jap_vows = JapaneseVowelsBuilder()
    #jap_vows.read_test_and_train_files()
    #print(jap_vows.get_training_data())
    five_and_dens = FiveBitAndDensityBuilder()
    five_and_dens.get_training_data()
#cifarB = CIFARBuilder()
#cifarB.get_cifar_data()
