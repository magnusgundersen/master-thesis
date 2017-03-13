__author__ = 'magnus'
import matplotlib.pyplot as plt


def visualize(list_of_states):
    list_of_states = list_of_states[::-1]
    width = len(list_of_states[0])
    gens = len(list_of_states)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pcolormesh(list_of_states, cmap="Greys")
    ax.set_xlim(0, width)
    ax.set_ylim(0, gens + 1)
    ax.set_title("CA simulation")
    #ax.set_axis_off()
    ax.set_aspect("equal")

    plt.show()

def visualize_multiple_reservoirs(list_of_reservoirs):
    lines = [[] for _ in range(len(list_of_reservoirs[0]))]
    for reservoir in list_of_reservoirs:
        print("reservoir length:" + str(len(reservoir)))
        for i in range(len(reservoir)):

            lines[i].extend(reservoir[i])



    list_of_states = lines[::-1]
    visualize(lines)

