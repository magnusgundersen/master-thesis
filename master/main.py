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

#import logger.logger as log

import reservoircomputing.rc as rc
__author__ = 'magnus'

def main(argv):
    print(argv)
    if "gui" in argv:
        print("Running gui version")
        print("NOT IMPLEMENTED")


    p = project.Project()  # Project with all implemented experiments



    #print(p.majority_task())
    #print(p.n_bit_task())
    print(p.img_clf_task())
    #print(p.run_bye_experiements())
    #print(p.run_yil_experiment_IR())
    #print(p.run_yil_experiment_RNN_distractor())
    #print(p.run_mg_experiments())


if __name__ == "__main__":
    main(sys.argv[1:])


