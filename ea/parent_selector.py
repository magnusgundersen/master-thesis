import random
import numpy as np


# "interface"
class ParentSelector:
    def produce_next_generation_children(self, adult_pool, number_of_children=0):
        """

        :param number_of_children:
        :return:
        """
        raise NotImplementedError

class TournamentSelector(ParentSelector):
    def __init__(self, tournament_size = 5):
        self.tournament_size = tournament_size
        self.probability_distribution = [0.7, 0.20, 0.09, 0.01]

    def produce_next_generation_children(self, adult_pool, number_of_children=0):
        # each parent pair is given two children.
        if number_of_children == 0:
            number_of_children = len(adult_pool)
        new_generation = []
        for i in range(number_of_children):
            tournament_one_members = np.random.choice(adult_pool, self.tournament_size)
            print(tournament_one_members)
            tournament_two_members = np.random.choice(adult_pool, self.tournament_size)
            print(tournament_two_members)

        return adult_pool





class RouletteSelector(ParentSelector):

    def produce_children(self, number_of_children, parent_roulette, mutation_rate=0.4, crossover_rate=0.01):
        children = []

        for i in range(int(number_of_children/2)):
            parent_1, parent_2 = self.pick_parents(parent_roulette)
            child_1 = parent_1.reproduce(parent_2.genotype)
            child_2 = parent_2.reproduce(parent_1.genotype)
            children.append(child_1)
            children.append(child_2)

        return children

    def pick_parents(self, roulette):
        parent_1_value = random.random()
        parent_1 = None
        parent_2 = None

        for (fitnessKeyStart, fitnessKeyStop) in roulette:  # TODO: revise if you need to use roulette.items()
            # print(fitnessKeyStart)
            if fitnessKeyStart < parent_1_value and fitnessKeyStop > parent_1_value:
                parent_1 = roulette[(fitnessKeyStart, fitnessKeyStop)]

        if parent_1 is None:
            print("ERROR. COULD NOT FIND FIRST PARENT")
            print("Roulette: " + str(roulette))
            raise ValueError

        while parent_2 is None:
            parent_2_value = random.random()

            for (fitnessKeyStart, fitnessKeyStop) in roulette:  # TODO: revise if you need to use roulette.items()
                if fitnessKeyStart<=parent_2_value and fitnessKeyStop>parent_2_value:
                    individual = roulette[(fitnessKeyStart, fitnessKeyStop)]
                    if individual == parent_1:
                        break
                    else:
                        parent_2 = individual
        # print("[EvoAlg] Parent_selection:pick_parents:" + str(parent_1.__class__))
        return parent_1, parent_2


    def get_total_fitness(self, pool):
        # Get the total fitness
        total_fitness = 0
        for individual in pool:
            total_fitness += individual.fitness
        return total_fitness

    # making a dict with all the individuals with the proportion of the roulette wheel that they deserve

    def produce_next_generation_children(self, adult_pool, number_of_children=0):
        self.adult_pool = adult_pool
        if number_of_children == 0:
            number_of_children = len(adult_pool)
        roulette_wheel = {}

        if not adult_pool:
            return []

        if self.parent_selection_type == "fitness proportionate":
            """
             we play the roulette with the fitness
            """
            current_point_on_roulette = 0
            total_fitness = self.get_total_fitness(self.adult_pool)
            for individual in self.adult_pool:
                fitness_part = individual.fitness/(total_fitness)
                roulette_wheel[(current_point_on_roulette, current_point_on_roulette+fitness_part)] = individual
                current_point_on_roulette += fitness_part
            # TODO: what if 1?

        else:
            print("ERROR: Could not understand parent selection type: " + self.parent_selection_type)
        return self.produce_children(number_of_children, roulette_wheel, 0.05, 0.3)