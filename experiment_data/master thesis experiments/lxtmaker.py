import json
import numpy as np
import matplotlib.pyplot as plt

with open("mass test 5 bit_thesis_Run/full_plotconfig.json","r") as f:
    data = json.load(f)
    best_to_worst = sorted(data, key=data.get, reverse=True)
    sorted_array = []
    for rule in best_to_worst:
        score = data.get(rule)[0]
        sorted_array.append([int(rule[5:]),int(score)])

    sorted_array = np.array(sorted_array, dtype="uint16")
    print(sorted_array.shape)
    print(sorted_array)

    np.savetxt("test.csv",sorted_array)


with open("mass test 20 bit thesis run/full_plotconfig.json","r") as f:
    data = json.load(f)
    best_to_worst = sorted(data, key=data.get, reverse=True)
    for rule in best_to_worst:
        score = data.get(rule)[0]
        if score != 0:
            #print(rule + ": "+str(data.get(rule)))
            pass


