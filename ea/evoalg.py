import operator
import random
from ea.individual import Individual
import ea.adult_selector as adlt_sel
import ea.parent_selector as parn_sel
import numpy as np

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

        parent_selector = parn_sel.TournamentSelector()
        adult_selector = adlt_sel.AdultSelector("full_and_elitism")



        while not problem.fitness_threshold(current_generation) \
                and problem.max_number_of_generations >= generation_number:
            new_gen = self.run_ea_step(current_generation, parent_selector, adult_selector, problem)
            current_generation = new_gen
            best_ind = new_gen[0]
            for ind in new_gen:
                if ind.fitness>best_ind.fitness:
                    best_ind = ind
            print(best_ind)
            fitness_values = np.array([ind.fitness for ind in new_gen])
            print("Mean of fitnesses: " + str(np.mean(fitness_values)))
            print("Std of fitnesses : " + str(np.std(fitness_values)))
            print()

            generation_number += 1


        return best_ind

    def run_ea_step(self, current_generation, parent_selector, adult_selector, ea_problem):



        children = parent_selector.produce_next_generation_children(current_generation)


        parallel = True
        if parallel:

            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list

            with multiprocessing.Pool(7) as p:
                children = p.starmap(develop_and_test, [(children[i], ea_problem) for i in range(len(children))])

        else:
            for child in children:
                child.develop()
                ea_problem.test_fitness(child)

        new_generation = adult_selector.run_adult_selection(current_generation, children)


        return new_generation





class EAProblem:
    def __init__(self):
        self.max_number_of_generations = 100000

    def fitness_threshold(self, *args):
        return False

    def get_initial_population(self):
        raise NotImplementedError("[EAProblem] Please implement function: get_initial_population()")

    def test_fitness(self, population):
        raise NotImplementedError()
