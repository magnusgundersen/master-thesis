# Creates a ReCAProblem from the dataset.
import pickle
import numpy as np3
import os

from PIL import Image
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
            dataset = []
            with open(file_location+"/en-de.data", "r") as f:
                content = f.readlines()
                training_set = []
                for line in content:
                    if line == "\n":
                        dataset.append(training_set)
                        training_set = []
                    else:
                        _input, _output = line.split(" ")
                        training_set.append(([int(number) for number in _input],_output[0:-1]))  # class is text
            return dataset
        else:
            raise ValueError("Language not implemented: " + str(self.language))

    def get_testing_data(self):
        return []

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
        from_lines = []
        file_location = os.path.dirname(os.path.realpath(__file__))
        with open(file_location+"/translation/" + "europarl-v7."+language+"-en."+language, "r", encoding='utf8') as f:
            content = f.readlines()
            for line in content:
                if line == "\n":
                    continue
                from_lines.append(line)

        to_lines = []
        with open(file_location+"/translation/" + "europarl-v7." + language + "-en." + "en", "r", encoding='utf8') as f:
            content = f.readlines()
            for line in content:
                if line == "\n":
                    continue
                to_lines.append(line)
        return (from_lines, to_lines)

    def generate_translation_data(self):

        eng_lines = []
        ger_lines = []

        german_to_english = (self.read_translation_files("de"))
        ger_lines = german_to_english[0]
        eng_lines = german_to_english[1]

        examples = 1000
        txt_bin_data = ""
        for i in range(examples):
            txt_bin_data += (self.create_bin_data(eng_lines[i], self.english_allowed, ger_lines[i], self.german_allowed))
            txt_bin_data += "\n"
        with open("en-de.data", "w+") as f:
            f.write(txt_bin_data)

    def create_bin_data(self, source_sentence, source_alphabet, target_sentence, target_alphabet):
        data_file = ""
        #print("Source sentence: " + str(source_sentence), end="")
        #print("target sentence: " + str(target_sentence), end="")
        for character in source_sentence:
            for i in range(len(source_alphabet)):
                if source_alphabet[i] == character.lower():
                    data_file += "1"
                    continue
                data_file += "0"

            data_file += " "

            for i in range(len(target_alphabet)):
                data_file += "0"
            data_file += "1"  # Wait signal
            data_file += "0"  # End of sentence signal
            data_file += "\n"


        for character in target_sentence:
            for i in range(len(source_alphabet)):
                data_file += "0"  # End of source sentence

            data_file += " "
            for i in range(len(target_alphabet)):
                if target_alphabet[i] == character.lower():
                    data_file += "1"
                    continue
                data_file += "0"
            data_file += "0"
            data_file += "0"
            data_file += "\n"

        self.prod_end_signal = "0"*len(target_alphabet)+ "0"+"1"+"\n"  # End of sentence
        data_file += "0"*len(source_alphabet) + " " + self.prod_end_signal


        return data_file


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
    def __init__(self):
        self.dist_period = 10
        self.no_training_ex = 32
        self.no_testing_ex = 32

    def get_training_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

        dataset = []
        with open(file_location+"/5bit/5_bit_" + str(self.dist_period) + "_dist_32", "r") as f:
            content = f.readlines()
            training_set = []
            for line in content:
                if line == "\n":
                    dataset.append(training_set)
                    training_set = []
                else:
                    _input, _output = line.split(" ")
                    training_set.append(([int(number) for number in _input],_output[0:-1]))  # class is text
        return dataset

    def get_testing_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

        dataset = []
        with open(file_location+"/5bit/5_bit_" + str(self.dist_period) + "_dist_32", "r") as f:
            content = f.readlines()
            training_set = []
            for line in content:
                if line == "\n":
                    dataset.append(training_set)
                    training_set = []
                else:
                    _input, _output = line.split(" ")
                    training_set.append(([int(number) for number in _input],_output[0:-1]))  # class is text
        return dataset



class TwentyBitBuilder:
    def __init__(self):
        self.dist_period = 10
        self.no_training_ex = 500
        self.no_testing_ex = 100

    def get_training_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

        dataset = []
        with open(file_location+"/20bit/20_bit_train_" + str(self.dist_period) + "_dist_" + str(self.no_training_ex), "r") as f:
            content = f.readlines()
            training_set = []
            for line in content:
                if line == "\n":
                    dataset.append(training_set)
                    training_set = []
                else:
                    _input, _output = line.split(" ")
                    training_set.append(([int(number) for number in _input],_output[0:-1]))  # class is text
        return dataset

    def get_testing_data(self):
        file_location = os.path.dirname(os.path.realpath(__file__))

        dataset = []
        with open(file_location+"/20bit/20_bit_test_" + str(self.dist_period) + "_dist_"+str(self.no_testing_ex), "r") as f:
            content = f.readlines()
            training_set = []
            for line in content:
                if line == "\n":
                    dataset.append(training_set)
                    training_set = []
                else:
                    _input, _output = line.split(" ")
                    training_set.append(([int(number) for number in _input],_output[0:-1]))  # class is text
        return dataset

if __name__ == "__main__":
    translator = TranslationBuilder()
    translator.generate_translation_data()


#cifarB = CIFARBuilder()
#cifarB.get_cifar_data()
