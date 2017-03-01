import operator
import random
from ea.individual import Individual

import multiprocessing

def develop_and_test(individual, problem):
    individual.develop()
    problem.test_fitness(individual)
    return individual


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
        parallel = True

        if parallel:


            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list

            with multiprocessing.Pool(7) as p:
                initial_population = p.starmap(develop_and_test, [(initial_population[i], problem) for i in range(len(initial_population))])

        else:
            for individual in initial_population:
                individual.develop()
                problem.test_fitness(individual)

        generation_number = 0
        current_generation = initial_population
        print()
        print("Current gen:")
        print(current_generation)
        print()

        parent_selector = self.ParentSelector("fitness proportionate")
        adult_selector = self.AdultSelector("mixing")



        while not problem.fitness_threshold(current_generation) \
                and problem.max_number_of_generations >= generation_number:
            new_gen = self.run_ea_step(current_generation, parent_selector, adult_selector, problem)
            current_generation = new_gen
            best_ind = new_gen[0]
            for ind in new_gen:
                if ind.fitness>best_ind.fitness:
                    best_ind = ind
            print(best_ind)
            print(new_gen)

            generation_number += 1


        return best_ind

    def run_ea_step(self, current_generation, parent_selector, adult_selector, ea_problem):
        children = parent_selector.run_parent_selection(current_generation)
        parallel = True
        if parallel:

            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list

            with multiprocessing.Pool(8) as p:
                children = p.starmap(develop_and_test, [(children[i], ea_problem) for i in range(len(children))])

        else:
            for child in children:
                child.develop()
                ea_problem.test_fitness(child)

        new_generation = adult_selector.run_adult_selection(current_generation, children)


        return new_generation

    class ParentSelector:
        def __init__(self, parent_selection_type):
            self.adult_pool = []
            self.parent_selection_type = parent_selection_type

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

        def adult_selector(self, pop_size, population):
            # does selection based on fitness
            #print("old pop: " + str(population))
            new_pop = []
            fitness_dict = {}
            for individual in population:
                fitness_dict[individual] = individual.fitness
            sorted_fitness = sorted(fitness_dict.items(), key=operator.itemgetter(1))[::-1]  # reverse for best first

            i = 0
            for (ind,fit) in sorted_fitness:
                if i == pop_size:
                    break
                new_pop.append(ind)
                i += 1
            return new_pop

        def run_adult_selection(self, old_generation, new_adult_candidates):
            """
            Choose who will join the adult pool.
            Can be: mixing
                    full
            :return: the new generation, after selection
            """
            adult_pool_size = len(old_generation)
            new_adult_pool = []

            #print("Old gen: " + str(old_generation))
            #print("candita: " + str(new_adult_candidates))

            if(self.selection_type=="full"):
                new_adult_pool.extend(new_adult_candidates)

            elif (self.selection_type == "mixing"):
                new_adult_candidates.extend(old_generation)
                new_adult_pool.extend(self.adult_selector(adult_pool_size, new_adult_candidates))

            else:
                print("Adult selection currently not implemented: " + self.selection_type)
            return new_adult_pool




class EAProblem:
    def __init__(self):
        self.max_number_of_generations = 100000

    def fitness_threshold(self, *args):
        return False

    def get_initial_population(self):
        raise NotImplementedError("[EAProblem] Please implement function: get_initial_population()")

    def test_fitness(self, population):
        raise NotImplementedError()
