# Creates a ReCAProblem from the dataset.

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









translator = TranslationBuilder()
translator.open_all_translation()
