import ea.individual as ind
import ea.evoalg as evoalg
import random
# testing
class BinVectorGeno(ind.Genotype):
    def __init__(self, parent_genotype_one, parent_genotype_two):
        super().__init__(parent_genotype_one, parent_genotype_two)

    def init_first_genotype(self):
        #print("First genotype!")
        self.bitstring = [random.choice([0,1]) for _ in range(20)]

    def get_representation(self):
        pass

    def reproduce(self, other_genotype, crossover_rate=0.5, mutation_rate=0.1):
        self.bitstring = self.bitstring[:7] + other_genotype.bitstring[7:]

        random_number = random.random()
        if random_number<mutation_rate:
            bitflip = random.randint(0,19)
            self.bitstring[bitflip] = 0 if self.bitstring[bitflip] == 1 else 1

class BinVectorPheno(ind.Phenotype):
    def __init__(self, genotype):
        super().__init__(genotype)
        self.bitstring = genotype.bitstring

    def develop_from_genotype(self):
        pass


class BinVectorInd(ind.Individual):
    def __init__(self, parent_genotype_one=None, parent_genotype_two=None):
        super().__init__(parent_genotype_one, parent_genotype_two)
        self.genotype = BinVectorGeno(parent_genotype_one, parent_genotype_two)
        self.phenotype = None



    def develop(self):
        self.phenotype = BinVectorPheno(self.genotype)

    def reproduce(self, other_parent_genotype):
        child = BinVectorInd(other_parent_genotype)
        return child

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()

class BINProblem(evoalg.EAProblem):
    def __init__(self):
        super().__init__()

    def test_fitness(self, individual):
        bitstring = individual.phenotype.bitstring
        individual.fitness = 1 if bitstring.count(1) == 0 else bitstring.count(1)
        return individual.fitness

    def get_initial_population(self):
        return [BinVectorInd() for _ in range(10000)]

if __name__ == "__main__":
    binprob = BINProblem()
    ea = evoalg.EA()

    ea.solve(binprob)
