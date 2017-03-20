import operator
import random
from ea.individual import Individual
import ea.adult_selector as adlt_sel
import ea.parent_selector as parn_sel
import numpy as np
import sys
import multiprocessing
import os
import pickle
import time



# Parallel-config
parallel = True
if parallel:
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 20:
        threads_to_be_used = 7  # 4 not cloging the machine
    else:
        threads_to_be_used = 20


# Workers:
def develop_and_test(individual, problem):
    individual.develop()
    individual = problem.test_fitness(individual)
    return individual

# Classes:
class EA:
    def __init__(self, crossover_rate=0.4, mutation_rate=0.15):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def solve(self, problem, saved_state=False):
        print("Running EA for problem: " + str(problem))

        current_time = time.time()
        if not saved_state:
            initial_population = problem.get_initial_population()
            parent_selector = parn_sel.TournamentSelector()
            adult_selector = adlt_sel.AdultSelector("full_and_elitism")
            generation_number = 0
            print("Starting new EA")
            if parallel:  # Global variable
                with multiprocessing.Pool(threads_to_be_used) as p:
                    initial_population = p.starmap(develop_and_test, [[initial_population[i], problem] for i in range(len(initial_population))])
                    #initial_population = p.starmap(develop_and_test_hacked, [[initial_population[i]] for i in range(len(initial_population))])
            else:
                for individual in initial_population:
                    individual.develop()
                    problem.test_fitness(individual)
            ea_output = EAOutput()
            ea_output.add_generation(initial_population)
        else:
            saved_state = self.load_ea_state()
            initial_population = saved_state["current_generation"]
            parent_selector = saved_state["parent_selector"]
            adult_selector = saved_state["adult_selector"]
            generation_number = saved_state["generation_number"]
            ea_output = saved_state["ea_output"]



        current_generation = initial_population

        best_ind = current_generation[0]
        for ind in current_generation:
            if ind.fitness > best_ind.fitness:
                best_ind = ind

        while not problem.fitness_threshold(best_ind.fitness) \
                and problem.max_number_of_generations > generation_number:


            sys.stdout.flush()
            sys.stdout.write("\r"+"Current generation: " + str(generation_number) +
                             " --Best fitness: " + str(ea_output.best_individuals_per_gen[-1].fitness) +
                             " ±"+ str(ea_output.best_individuals_per_gen[-1].fitness_std) +
                             " --Mean fitness:  " + str((ea_output.mean_fitness_per_gen[-1])) +
                             " --Time usage : m: " + str((time.time() - current_time)//60) +
                                            " s: " + str(int((time.time() - current_time)%60))
                             )
            current_time = time.time()

            if (generation_number % 10) == 0:  # Every 10th gen gets a new line
                print("\n", end="")

            new_gen = self.run_ea_step(current_generation, parent_selector, adult_selector, problem)
            current_generation = new_gen
            generation_number += 1

            ea_output.add_generation(current_generation)

            self.save_ea_state(current_generation, parent_selector, adult_selector, generation_number, ea_output)

            best_ind = ea_output.best_individual

        print("\n", end="")
        return ea_output

    @staticmethod
    def save_ea_state(current_generation, parent_selector, adult_selector, generation_number, ea_output):
        file_location = os.path.dirname(os.path.realpath(__file__))+"/../experiment_data/ea_runs/"
        persistance_file = file_location + "state.ea"
        backup_file = file_location + "state_old.ea"
        saved_state = {}
        saved_state["current_generation"] = current_generation
        saved_state["parent_selector"] = parent_selector
        saved_state["adult_selector"] = adult_selector
        saved_state["generation_number"] = generation_number
        saved_state["ea_output"] = ea_output
        if os.path.isfile(persistance_file):
            if os.path.isfile(backup_file):
                os.remove(backup_file)
            os.rename(persistance_file, file_location+"state_old.ea")

        pickle.dump(saved_state, open(persistance_file, 'wb'))

    @staticmethod
    def load_ea_state(file_name="state.ea"):
        file_location = file_location = os.path.dirname(os.path.realpath(__file__))+"/../experiment_data/ea_runs/"
        state_data = pickle.load(open(file_location+file_name, "rb"))
        return state_data

    @staticmethod
    def run_ea_step(current_generation, parent_selector, adult_selector, ea_problem):

        children = parent_selector.produce_next_generation_children(current_generation)

        if parallel:
            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list
            with multiprocessing.Pool(threads_to_be_used) as p:
                children = p.starmap(develop_and_test, [[children[i], ea_problem] for i in range(len(children))])
                #children = p.starmap(develop_and_test_hacked, [[children[i]] for i in range(len(children))])

        else:  # sequential execution
            for child in children:
                child.develop()
                ea_problem.test_fitness(child)

        new_generation, elite_pop = adult_selector.run_adult_selection(current_generation, children)


        new_elite_pop = []
        restest_elite_pop = True
        if restest_elite_pop:
            if parallel:
                # Create a list of jobs and then iterate through
                # the number of processes appending each process to
                # the job list
                with multiprocessing.Pool(5) as p:
                    new_elite_pop = p.starmap(develop_and_test, [[elite_pop[i], ea_problem] for i in range(len(elite_pop))])
                    #new_elite_pop = p.starmap(develop_and_test_hacked, [[elite_pop[i]] for i in range(len(elite_pop))])

            else:  # sequential execution
                for ind in elite_pop:
                    ea_problem.test_fitness(ind)
        else:
            new_elite_pop = elite_pop
        new_generation = new_generation + new_elite_pop

        return new_generation


class EAProblem:
    def __init__(self):
        self.max_number_of_generations = 1000 # Default

    def fitness_threshold(self, *args):  # Some fitness-value such that the execution is stopped
        raise NotImplementedError()

    def get_initial_population(self):
        raise NotImplementedError("[EAProblem] Please implement function: get_initial_population()")

    def test_fitness(self, individual):
        raise NotImplementedError()

    def __str__(self):
        return "EA problem"


class EAOutput:
    def __init__(self):
        self.all_generations = []  # list of lists with all generations
        self.best_individuals_per_gen = []
        self.mean_fitness_per_gen = []
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
