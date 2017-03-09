import operator
import random
from ea.individual import Individual
import ea.adult_selector as adlt_sel
import ea.parent_selector as parn_sel
import numpy as np

import multiprocessing


# Parallel
parallel = True
threads_to_be_used = 8  # number of logical thread that the run concurrently
def develop_and_test(individual, problem):
    """
    Method for running the develop_and_test with multiprocessing
    :param individual:
    :param problem:
    :return:
    """
    individual.develop()
    problem.test_fitness(individual)
    return individual


class EA:
    def __init__(self, crossover_rate=0.5, mutation_rate=0.1):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def solve(self, problem):
        ea_output = EAOutput()

        initial_population = problem.get_initial_population()

        if parallel:  # Global variable
            with multiprocessing.Pool(threads_to_be_used) as p:
                initial_population = p.starmap(develop_and_test, [(initial_population[i], problem) for i in range(len(initial_population))])
        else:
            for individual in initial_population:
                individual.develop()
                problem.test_fitness(individual)

        generation_number = 0
        current_generation = initial_population

        parent_selector = parn_sel.TournamentSelector()
        adult_selector = adlt_sel.AdultSelector("full_and_elitism")

        best_ind = current_generation[0]
        for ind in current_generation:
            if ind.fitness > best_ind.fitness:
                best_ind = ind

        ea_output.add_generation(current_generation)
        while not problem.fitness_threshold(best_ind.fitness) \
                and problem.max_number_of_generations > generation_number:

            new_gen = self.run_ea_step(current_generation, parent_selector, adult_selector, problem)
            current_generation = new_gen

            ea_output.add_generation(current_generation)
            generation_number += 1


        return ea_output

    def run_ea_step(self, current_generation, parent_selector, adult_selector, ea_problem):

        children = parent_selector.produce_next_generation_children(current_generation)

        if parallel:
            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list
            with multiprocessing.Pool(threads_to_be_used) as p:
                children = p.starmap(develop_and_test, [(children[i], ea_problem) for i in range(len(children))])

        else:  # sequential execution
            for child in children:
                child.develop()
                ea_problem.test_fitness(child)

        new_generation = adult_selector.run_adult_selection(current_generation, children)

        return new_generation





class EAProblem:
    def __init__(self):
        self.max_number_of_generations = 25 # Default

    def fitness_threshold(self, *args):  # Some fitness-value such that the execution is stopped
        raise NotImplementedError()

    def get_initial_population(self):
        raise NotImplementedError("[EAProblem] Please implement function: get_initial_population()")

    def test_fitness(self, population):
        raise NotImplementedError()

class EAOutput:
    def __init__(self):
        self.all_generations = []  # list of lists with all generations
        self.best_individuals_per_gen = []
        self.mean_fitness_per_gen =  []
        self.std_per_gen = []
        self.best_individual = None

        self.printing = False

    def add_generation(self, generation):
        best_ind = generation[0]
        for ind in generation:
            if ind.fitness > best_ind.fitness:
                best_ind = ind

        self.all_generations.append(generation)
        self.best_individuals_per_gen.append(best_ind)
        if len(self.all_generations)>1:
            if best_ind.fitness > self.best_individual.fitness:
                self.best_individual = best_ind
        else:
            self.best_individual = best_ind

        fitness_values = np.array([ind.fitness for ind in generation])
        self.mean_fitness_per_gen.append(np.mean(fitness_values))
        self.std_per_gen.append((np.std(fitness_values)))
        if self.printing:
            print("Generation: " + str(len(self.all_generations)-1))
            print(best_ind)
            print("Mean of fitnesses: " + str(self.mean_fitness_per_gen[-1]))
            print("Std of fitnesses : " + str(self.std_per_gen[-1]))
            print()
