# Creates a ReCAProblem from the dataset.
import pickle
import numpy as np
import os
import pprint
import xml.etree.ElementTree as ET
import scipy.sparse as sprs
import scipy.io as sci_io
#from speech import Speech, SpeakerStatistics, SpeakerStatisticsCollection
from PIL import Image
#import Image
class TranslationBuilder:
    def __init__(self):
        """
        utf-8
        """
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
        file_location = os.path.dirname(os.path.realpath(__file__))
        if self.language == "german":
            dataset = []  #
            batches_location = file_location + "/translation/en-de/"
            lang_locations = batches_location + "/de"
            eng_locations = batches_location + "/en"
            lang_files = os.listdir(lang_locations)
            eng_files = os.listdir(eng_locations)

            #print(lang_files)
            number_of_examples = len(lang_files)
            number_of_examples = 5000

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

    def get_testing_data(self):
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
        return sentence

    def read_translation_files(self, language):
        file_location = os.path.dirname(os.path.realpath(__file__))
        usable_eng_lines = []
        usable_lang_lines = []

        with open(file_location+"/translation/txt/" + "europarl-v7."+language+"-en."+language, "r", encoding='utf8') as lang,\
                open(file_location+"/translation/txt/" + "europarl-v7." + language + "-en." + "en", "r", encoding='utf8') as eng:
            lang = lang.readlines()
            eng = eng.readlines()

            if len(lang) != len(eng):
                raise ValueError("Both languages must have the same number of lines!")

            breakoff = True
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
        sequence_length = len(source_sentence) + len(target_sentence) + 1  # : +1: prediction end signal
        source_array = np.zeros((sequence_length, len(source_alphabet)+1), dtype="uint8")  # + 1: end of source sentence
        target_array = np.zeros((sequence_length, len(target_alphabet)+1+1), dtype="uint8")  # + 1 +1 : wait and end of sentence

        # Create source and target array inputs/outputs for source sentence
        for i in range(len(source_sentence)):
            for j in range(len(source_alphabet)):
                character = source_alphabet[j]
                if source_alphabet[j] == character.lower():
                    source_array[i][j] = 1
                    break
            target_array[i][-2] = 1  # wait signal


        for i in range(len(source_sentence), len(source_sentence) + len(target_sentence)):
            source_array[i][-1] = 1  # source sentence ended, produce translated sentence signal
            for j in range(len(target_alphabet)):
                character = target_alphabet[j]
                if target_alphabet[j] == character.lower():
                    target_array[i][j] = 1
                    break


        ## prediction end signal:
        source_array[-1][-1] = 1
        target_array[-1][-1] = 1
        signal = np.zeros(len(target_array[-1]))
        signal[-1] = 1
        self.prediction_end_signal = signal # End of sentence

        #print(source_array.shape)
        #print(target_array.shape)

        return source_array, target_array



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
    def __init__(self, dist_period=200):
        self.dist_period = dist_period
        self.no_training_ex = 32
        self.no_testing_ex = 32

    def get_training_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

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
        file_location = os.path.dirname(os.path.realpath(__file__))

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
    def __init__(self):
        self.dist_period = 280
        self.no_training_ex = 120
        self.no_testing_ex = 100

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

if __name__ == "__main__":
    translator = TranslationBuilder()
    #translator.create_efficient_data_to_file()
    translator.generate_translation_data()

#cifarB = CIFARBuilder()
#cifarB.get_cifar_data()
