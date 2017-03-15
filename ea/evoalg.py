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

# REMOVE BELOW
import reca.reca_system as reCA
import experiment_data.data_interpreter as data_int
# STOP REMOVE


# Parallel-config
parallel = True
if parallel:
    cpu_count = multiprocessing.cpu_count()
    if cpu_count < 20:
        threads_to_be_used = 7  # 4 not cloging the machine
    else:
        threads_to_be_used = 20


def open_data_interpreter(type_of_interpreter, **kwargs):
    if type_of_interpreter == "europarl":
        return data_int.TranslationBuilder()

    elif type_of_interpreter == "5bit":
        distractor_period = kwargs.get("distractor_period") if kwargs.get('distractor_period') is not None else 10
        training_ex = kwargs.get("training_ex") if kwargs.get('training_ex') is not None else 32
        testing_ex = kwargs.get("testing_ex") if kwargs.get('testing_ex') is not None else 32
        return data_int.FiveBitBuilder(distractor_period, training_ex=training_ex, test_ex=testing_ex)

    elif type_of_interpreter == "20bit":
        return data_int.TwentyBitBuilder()

def develop_and_test_hacked(individual):
    """
    Method for running the develop_and_test with multiprocessing
    :param individual:
    :param problem:
    :return:
    """

    individual.develop()
    # NB: HACKED FOR CA EA
    before_tesing = time.time()
    fitness = 0
    #print("testing fitness")
    tests_per_ind = 4
    for _ in range(tests_per_ind):
        reCA_problem = reCA.ReCAProblem(open_data_interpreter("5bit", distractor_period=10, training_ex=32, testing_ex=16))
        reCA_config = reCA.ReCAConfig()

        reCA_rule_scheme = reCA.ReCAruleConfig(non_uniform_list=individual.phenotype.non_uniform_config)

        reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule_scheme, R=16, C=4, I=4,
                                              classifier="perceptron_sgd")
        reCA_system = reCA.ReCASystem()

        reCA_system.set_problem(reCA_problem)
        reCA_system.set_config(reCA_config)
        reCA_system.initialize_rc()
        reCA_system.tackle_ReCA_problem()

        reCA_out = reCA_system.test_on_problem()
        fitness += int((reCA_out.total_correct / len(reCA_out.all_test_examples)) * 1000)

    # pseudo_lambda = self.calculate_pseudo_lambda(individual.phenotype.non_uniform_config)
    # fitness += pseudo_lambda
    fitness = 1 if fitness == 0 else fitness // tests_per_ind
    individual.fitness = fitness

    making_sure_fitness = 0
    if fitness > 1000:
        print("\n" + "Investigating fitness! " + str(individual))
        making_sure_tests = 10
        for _ in range(making_sure_tests):
            data_interpreter = open_data_interpreter("5bit", distractor_period=10, training_ex=32, testing_ex=32)
            reCA_problem = reCA.ReCAProblem(data_interpreter)
            reCA_config = reCA.ReCAConfig()

            reCA_rule_scheme = reCA.ReCAruleConfig(non_uniform_list=individual.phenotype.non_uniform_config)

            reCA_config.set_random_mapping_config(ca_rule_scheme=reCA_rule_scheme, R=16, C=4, I=4,
                                                  classifier="perceptron_sgd")
            reCA_system = reCA.ReCASystem()

            reCA_system.set_problem(reCA_problem)
            reCA_system.set_config(reCA_config)
            reCA_system.initialize_rc()
            reCA_system.tackle_ReCA_problem()

            reCA_out = reCA_system.test_on_problem()
            making_sure_fitness += int((reCA_out.total_correct / len(reCA_out.all_test_examples)) * 1000)
        making_sure_fitness = making_sure_fitness // making_sure_tests
        individual.fitness = making_sure_fitness

    #print("Finished testing fitness. Time: " + str((time.time() - before_tesing)))

    return individual

def develop_and_test(individual, problem):
    individual.develop()
    individual = problem.test_fitness(individual)
    return individual


class EA:
    def __init__(self, crossover_rate=0.4, mutation_rate=0.15):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def solve(self, problem, saved_state=False):
        print("Running EA for problem: " + str(problem))

        if not saved_state:
            initial_population = problem.get_initial_population()
            parent_selector = parn_sel.TournamentSelector()
            adult_selector = adlt_sel.AdultSelector("full_and_elitism")
            generation_number = 0
            print("Not daved")
            if parallel:  # Global variable
                with multiprocessing.Pool(threads_to_be_used) as p:
                    #initial_population = p.starmap(develop_and_test, [[initial_population[i], problem] for i in range(len(initial_population))])
                    initial_population = p.starmap(develop_and_test_hacked, [[initial_population[i]] for i in range(len(initial_population))])
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
        current_time = time.time()
        while not problem.fitness_threshold(best_ind.fitness) \
                and problem.max_number_of_generations > generation_number:


            sys.stdout.flush()
            sys.stdout.write("\r"+"Current generation: " + str(generation_number) +
                             " --Best fitness: " + str(ea_output.best_individuals_per_gen[-1].fitness) +
                             " --Mean fitness:  " + str((ea_output.mean_fitness_per_gen[-1])) +
                             " --Time usage last gen: m: " + str((time.time() - current_time)//60) +
                                                    " s: " + str(int((time.time() - current_time)%60))
                             )
            current_time = time.time()

            if generation_number != 0 and (generation_number % 10) == 0:  # Every 10th gen gets a new line
                print("\n", end="")

            new_gen = self.run_ea_step(current_generation, parent_selector, adult_selector, problem)
            current_generation = new_gen
            generation_number += 1

            ea_output.add_generation(current_generation)

            self.save_ea_state(current_generation, parent_selector, adult_selector, generation_number, ea_output)

            best_ind = ea_output.best_individual

        print("\n", end="")
        return ea_output

    def save_ea_state(self, current_generation, parent_selector, adult_selector, generation_number, ea_output):
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

    def load_ea_state(self, file_name="state.ea"):
        file_location = file_location = os.path.dirname(os.path.realpath(__file__))+"/../experiment_data/ea_runs/"
        state_data = pickle.load(open(file_location+file_name, "rb"))
        return state_data

    def run_ea_step(self, current_generation, parent_selector, adult_selector, ea_problem):

        children = parent_selector.produce_next_generation_children(current_generation)

        if parallel:
            # Create a list of jobs and then iterate through
            # the number of processes appending each process to
            # the job list
            with multiprocessing.Pool(threads_to_be_used) as p:
                #children = p.starmap(develop_and_test, [[children[i], ea_problem] for i in range(len(children))])
                children = p.starmap(develop_and_test_hacked, [[children[i]] for i in range(len(children))])

        else:  # sequential execution
            for child in children:
                child.develop()
                ea_problem.test_fitness(child)

        new_generation, elite_pop = adult_selector.run_adult_selection(current_generation, children)


        new_elite_pop = []
        restest_elite_pop = False
        if restest_elite_pop:
            if parallel:
                # Create a list of jobs and then iterate through
                # the number of processes appending each process to
                # the job list
                with multiprocessing.Pool(5) as p:
                    new_elite_pop = p.starmap(develop_and_test_hacked, [(elite_pop[i]) for i in range(len(elite_pop))])

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

    def test_fitness(self, population):
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
