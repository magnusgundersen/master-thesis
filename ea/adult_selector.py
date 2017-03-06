import operator
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
