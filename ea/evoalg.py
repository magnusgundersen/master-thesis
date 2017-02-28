import operator
import random
from ea.individual import Individual

class EA:
    def __init__(self):
        self.initial_population = []
        self.crossover_rate = 0.5
        self.mutation_rate = 0.1
        self.current_generation = []
        self.new_generation = []
        self.population_size = 0


    def set_init_pop(self, initial_pop):
        self.new_generation = initial_pop
        self.population_size = len(initial_pop)

    def solve(self, problem):
        initial_population = problem.get_initial_population()

        for individual in initial_population:
            individual.develop()

        generation_number = 0
        current_generation = initial_population

        parent_selector = self.ParentSelector("fitness proportionate")
        adult_selector = self.AdultSelector("full")



        while not problem.fitness_threshold(current_generation) \
                and problem.max_number_of_generations >= generation_number:
            new_gen = self.run_ea_step(initial_population, parent_selector, adult_selector, problem)
            print(new_gen)
            generation_number += 1


        pass

    def run_ea_step(self, current_generation, parent_selector, adult_selector, ea_problem):

        children = parent_selector.run_parent_selection(current_generation)
        print(children)


        for child in children:
            child.develop()
            ea_problem.test_fitness(child)

        new_generation = adult_selector.run_adult_selection(current_generation, children)


        return new_generation

    def get_best_individual(self):
        best_individual = self.current_generation[0]
        # assert isinstance(best_individual, Individual)

        for individual in self.current_generation:
            if best_individual.getFitness() < individual.getFitness():
                best_individual = individual

        return best_individual

    def get_worst_individual(self):
        worst_individual = self.current_generation[0]
        # assert isinstance(best_individual, Individual)

        for individual in self.current_generation:
            if worst_individual.getFitness() > individual.getFitness():
                worst_individual = individual

        return worst_individual
    class ParentSelector:
        def __init__(self, parent_selection_type):
            self.adult_pool = []
            self.parent_selection_type = parent_selection_type

        def produce_children(self, number_of_children, parent_roulette, mutation_rate=0.4, crossover_rate=0.01):
            children = []

            for i in range(number_of_children):
                parent_1, parent_2 = self.pick_parents(parent_roulette)
                child = self.produce_child(parent_1, parent_2)
                # print("[EvoAlg] Parent selector: produce children: child type: " + str(child.__class__))
                children.append(child)
                # TODO: consider giving the same parent-pair two children.

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


        def produce_child(self, parent_one, parent_two):


            # assert isinstance(parent_one,Individual)
            # assert isinstance(parent_two,Individual)
            # """@:type : Individual """
            child = parent_one.reproduce(parent_one.genotype,
                                         parent_two.genotype)

            return child

        def get_total_fitness(self, pool):
            # Get the total fitness
            total_fitness = 0
            for individual in pool:
                total_fitness += individual.fitness
            return total_fitness

        # making a dict with all the individuals with the proportion of the roulette wheel that they deserve

        def run_parent_selection(self, adult_pool, number_of_children=0):
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
                    fitness_part = individual.fitness/(total_fitness)  # TODO: FITNESS
                    roulette_wheel[(current_point_on_roulette, current_point_on_roulette+fitness_part)] = individual
                    current_point_on_roulette += fitness_part
                # TODO: what if 1?
            else:
                print("ERROR: Could not understand parent selection type: " + self.parent_selection_type)
            return self.produce_children(number_of_children, roulette_wheel, 0.05, 0.3)

    class AdultSelector:
        def __init__(self, selection_type,):
            self.selection_type = selection_type

            pass


        def elitism(self, number_of_lucky_ones, current_generation, new_generation):

            if not current_generation:
                return new_generation

            if number_of_lucky_ones == 0:
                return new_generation
            lucky_ones = []

            for i in range(number_of_lucky_ones):
                individual = self.get_best_individual()
                x = 0
                for j in range(len(current_generation)):
                    if individual == current_generation[i]:
                        lucky_ones.append(individual)
                        x=i
                        break
                current_generation.pop(x)
                new_generation.pop(0)
            new_generation.extend(lucky_ones)

            return new_generation

        def adult_selector(self, pop_size, population):
            # does selection based on fitness
            # dict with sorted values?
            new_pop = []
            fitness_dict = {}
            for individual in population:
                fitness_dict[individual] = individual.getFitness()
            sorted_fitness = sorted(fitness_dict.items(), key=operator.itemgetter(1))
            reversed_fitness = sorted_fitness[::-1]
            # print("REversed fitness: " + str(reversed_fitness))
            i = 0
            for (ind,fit) in sorted_fitness:
                if i == pop_size:
                    break
                new_pop.append(ind)

            return new_pop


        def run_adult_selection(self, old_generation, new_adult_candidates):
            """
            Choose who will join the adult pool.
            Can be: overprod
                    mixing
                    full
            :return: the new generation, after selection
            """
            adult_pool_size = len(old_generation)
            new_adult_pool = []
            if(self.selection_type=="full"):
                new_adult_pool.extend(new_adult_candidates)

            #elif (self.selection_type == "overprod"):
            #
             #   extra_children_size = int(adult_pool_size*0.2)
              #  extra_children = parent_selector.run_parent_selection(old_generation, extra_children_size)
               # new_adult_candidates.extend(extra_children)
                #new_adult_pool.extend(self.adult_selector(adult_pool_size, new_adult_candidates))



            elif (self.selection_type == "mixing"):
                new_adult_candidates.extend(old_generation)
                new_adult_pool.extend(self.adult_selector(adult_pool_size, new_adult_candidates))





            else:
                print("Adult selection currently not implemented: " + self.selection_type)
            return new_adult_pool




class EAProblem:
    def __init__(self):
        self.max_number_of_generations = 10

    def fitness_threshold(self, *args):
        return False

    def get_initial_population(self):
        raise NotImplementedError("[EAProblem] Please implement function: get_initial_population()")



    def test_fitness(self, population):
        raise NotImplementedError()
