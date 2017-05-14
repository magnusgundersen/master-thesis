"""
Main-file for executing the ReCA framework written by Magnus Gundersen

System consists of:
Cellular automata simulator.
Classifiers
Reservoir computation framework in which the CA may be used as a reservoir

gui-package: Tentative GUI-package which is proposed used to visualize the computations
tests-package: Tentative package for testing the various parts of the system. e.g. the rules of the CA.

"""
import sys
#import reservoir.ca as ca
#import classifier.skl_svm as svmclf
#import random
import master.project as project
import time

#import logger.logger as log

import reservoircomputing.rc as rc
__author__ = 'magnus'

def main(argv):
    if "gui" in argv:
        print("Running gui version")
        print("NOT IMPLEMENTED")


    p = project.Project()  # Project with all implemented experiments


    #before = time.time()

    #p.five_bit_task()
    #p.twenty_bit_task()
    #p.japanese_vowels()


    #### Evolve ####
    #p.evolve_ca_five_bit()
    #p.evolve_ca_twenty_bit()

    #p.evolve_ca_five_bit_and_density()
    #p.evolve_ca_jap_vowels()


    #p.five_bit_density_task()
    #p.square_root_sequence_task()
    #print(project.test_all_rules())
    #print(p.europarl_translation_task())
    #p.sequence_to_sequence_synth_task()
    #print(p.classifier_testing())
    #p.test_all_rules()

    #### mass testing ####
    #p.mass_test_five_bit_task()
    p.mass_test_5bit_density_task()

    #### Misc ####
    #p.run_ca_simulation()

if __name__ == "__main__":
    main(sys.argv[1:])


