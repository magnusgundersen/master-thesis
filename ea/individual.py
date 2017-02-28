import random


class Individual:
    def __init__(self, parent_genotype_one=None, parent_genotype_two=None):
        self.name = "Individual" + " " + str(random.randint(0,100000))
        self.fitness = 1

    def develop(self):
        """
        Developing the Phenotype from the given Genotype
        """
        raise NotImplementedError()

    def __str__(self):
        return self.name + " (" + str(self.fitness) + ")"

    def __repr__(self):
        return self.__str__()



class Genotype:
    def __init__(self, parent_genotype_one, parent_genotype_two):
        if parent_genotype_one is not None and parent_genotype_two is not None:
            self.reproduce(parent_genotype_one, parent_genotype_two)
        else:
            self.init_first_genotype()

        self.representation = None

    def get_representation(self):
        raise NotImplementedError("[Genotype] Please implement this method: __get_representation")

    def init_first_genotype(self):
        """
        this is for initializing the initial population
        :return:
        """
        raise NotImplementedError("[Genotype] Please implement this method: init_first_genotype")

    def reproduce(self, parent1, parent2, crossover_rate=0.4, mutation_rate=0.01):
        """
        this is the reproduction function. Shall deal with the crossover, and mutation and the joining of two genotypes
        :param parent1:
        :param parent2:
        :param crossover_rate:
        :param mutation_rate:
        :return:
        """
        raise NotImplementedError("[Genotype] Please implement this method: reproduce")



    pass


class Phenotype:
    def __init__(self, genotype):
        self.genotype = genotype

    def develop_from_genotype(self):
        raise NotImplementedError("[Phenotype] Please implement this method: __develop_from_phenotype")



