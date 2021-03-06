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
    #p.evolve_synthetic_seq_to_seq()

    #p.evolve_sqrt_seq()
    #for i in range(10):
    #    p.evolve_ca_twenty_bit()
    #
    for i in range(14):
        try:
            #p.evolve_ca_jap_vowels()
            #p.evolve_sqrt_seq()
            pass

        except:
            print("Evolve  failed. Continuing..")
            continue


    #p.five_bit_density_task()
    #p.square_root_sequence_task()
    #print(project.test_all_rules())
    #p.europarl_translation_task()
    #p.sequence_to_sequence_synth_task()
    #print(p.classifier_testing())
    #p.test_all_rules()

    #### mass testing ####
    p.mass_test_five_bit_task()
    #p.mass_test_5bit_density_task()
    #p.mass_test_twenty_bit_task()
    #p.mass_test_japanese_vowels_task()
    #p.mass_test_sqrt_seq_task()

    #### Misc ####
    #p.run_ca_simulation()
    #p.run_reca_sample_simulation()

if __name__ == "__main__":
    main(sys.argv[1:])


