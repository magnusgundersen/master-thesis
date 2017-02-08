# Creates a ReCAProblem from the dataset.
import pickle
import numpy as np

from PIL import Image
class TranslationBuilder:
    def __init__(self):
        """
        utf-8
        """
        pass

    def read_translation_files(self, language):
        from_lines = []
        with open("translation/" + "europarl-v7."+language+"-en."+language, "r", encoding='utf8') as f:
            content = f.readlines()
            for line in content:
                if line == "\n":
                    continue
                from_lines.append(line)

        to_lines = []
        with open("translation/" + "europarl-v7." + language + "-en." + "en", "r", encoding='utf8') as f:
            content = f.readlines()
            for line in content:
                if line == "\n":
                    continue
                to_lines.append(line)
        return (from_lines, to_lines)


    def open_all_translation(self):
        print("Translation")
        english_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                            "s", "t", "u", "v", "w", "x", "y", "z"]

        german_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                           "s", "t", "u", "v", "w", "x", "y", "z", "ä", "ö", "ü"]

        french_alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
                           "s", "t", "u", "v", "w", "x", "y", "z"]

        common_signs = [" ", ".", ",", ":", ";", "-", "!", "?", "'", "(", ")", "%", "/", '"', "'", "+"]
        arabic_numbers = [str(i) for i in range(10)]

        english_allowed = english_alphabet + common_signs + arabic_numbers
        german_allowed = german_alphabet + common_signs + arabic_numbers
        eng_lines = []
        ger_lines = []

        german_to_english = (self.read_translation_files("de"))
        ger_lines = german_to_english[0]
        eng_lines = german_to_english[1]

        print("number of german sentences: " + str(len(ger_lines)))
        print("number of english sentences: " + str(len(eng_lines)))

        print("size of english allowed alphabet: " + str(len(english_allowed)))
        self.create_bin_data(eng_lines[0], english_allowed, ger_lines[0], german_allowed, 1)

    def create_bin_data(self, source_sentence, source_alphabet, target_sentence, target_alphabet, computational_period):
        data_file = ""
        print("Source sentence: " + str(source_sentence), end="")
        print("target sentence: " + str(target_sentence), end="")
        for character in source_sentence:
            for i in range(len(source_alphabet)):
                if source_alphabet[i] == character.lower():
                    data_file += "1"
                    continue
                data_file += "0"

            data_file += " "

            for i in range(len(target_alphabet)):
                data_file += "0"
            data_file += "1"

            data_file += "\n"

        for period in range(computational_period):
            data_file += "0"*len(source_alphabet) + " " + "0"*len(target_alphabet)+ "1"+"\n"

        for character in target_sentence:
            for i in range(len(source_alphabet)):
                data_file += "0"

            data_file += " "
            for i in range(len(target_alphabet)):
                if target_alphabet[i] == character.lower():
                    data_file += "1"
                    continue
                data_file += "0"
            data_file += "0"




            data_file += "\n"


        print(data_file)


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
            if chn > 120:  # THRESHOLD
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







#translator = TranslationBuilder()
#translator.open_all_translation()


cifarB = CIFARBuilder()
cifarB.get_cifar_data()
