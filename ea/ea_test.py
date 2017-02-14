import ea.individual as ind
import ea.evoalg as evoalg

# testing
class BinVectorInd(ind.Individual):

    def test_fitness(self, *args):
        pass


init_pop = [BinVectorInd(), BinVectorInd(), BinVectorInd(), BinVectorInd()]
ea = evoalg.EA()
ea.set_init_pop(init_pop)
ea.run_ea_alg(init_pop)

