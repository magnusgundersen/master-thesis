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
    #p.five_bit_density_task()
    p.japanese_vowels()
    #p.evolve_and_test_non_uni_ca_jap_vowls()
    #print(project.test_all_rules())
    #print(p.evolve_and_test_non_uni_ca())
    #print(p.europarl_translation_task())
    #p.sequence_to_sequence_synth_task()
    #print(p.twenty_bit_task())
    #print(p.classifier_testing())
    #p.test_rules([90],[[141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 141, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 210, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]])
    #print("Time used: " + str(time.time()-before))


if __name__ == "__main__":
    main(sys.argv[1:])


