
import os
import pickle
import json
import matplotlib.pyplot as plt
import tabulate as tabulate

all_items = os.listdir()
folders = []
for folder in all_items:
    if os.path.isdir(folder):
        folders.append(folder)

ea_runs_file_names = {}
ea_info_files_names = {}
for folder in folders:
    all_items = os.listdir(folder)
    for item in all_items:
        #print(item)
        if item.endswith("report.json"):
            ea_runs_file_names[int(folder)] = folder+"/"+item

        if item.endswith("JSON.json"):
            ea_info_files_names[int(folder)] = folder + "/" + item
ea_runs = {}
best_ind_info= {}
for ea_run_number in ea_runs_file_names.keys():
    ea_run_name = ea_runs_file_names.get(ea_run_number)
    ea_info_files_name = ea_info_files_names.get(ea_run_number)
    ea_run = json.load(open(ea_run_name, "r"))
    ea_runs[int(ea_run_number)] = ea_run
    best_ind_info[int(ea_run_number)] = json.load(open(ea_info_files_name,"r"))



names = [str(i) for i in ea_runs.keys()]
generations_to_include = 1000 # max 1000


plot_data = [ea_runs.get(int(ea_run)).get("best_fitness")[:generations_to_include] for ea_run in names]

max_generations = max([len(fitness_list) for fitness_list in plot_data])


fig = plt.figure() # figsize=(30, 24)

ax1 = fig.add_subplot(1, 1, 1)
ax1.set_ylim([0, 1250])
ax1.set_xlim([0, max_generations+3])
ax1.set_title("5-bit EA") # , size=32
name = "full graph"
plot_colors = ["r", "g", "b", "y", "m", "k", "c"]
plot_colors = ["#023FA5","#7D87B9","#BEC1D4",
               "#4A6FE3","#8595E1","#B5BBE3",
               "#11C638","#8DD593","#C6DEC7",
               "#0FCFC0","#9CDED6","#D5EAE7"]

plot_colors= [
        "#002b36",
        "#073642",
        "#586e75",
        "#657b83",
        "#839496",
        "#93a1a1",
        "#b58900",
        "#cb4b16",
        "#dc322f",
        "#d33682",
        "#6c71c4",
        "#268bd2",
        "#2aa198",
        "#859900",
]

#plot_colors = ["black"]
#plot_colors = ["(1,2,2)"]
i = 0

for data in plot_data:  # For nice legend   plot_data
    plots = []

    # Make an example plot with two subplots...
    clf_plot1 = ax1.plot(range(len(data)), data, plot_colors[i % len(plot_colors)], label=str(names[i]))

    #ax1.plot(range(generations), data, plot_colors[i % len(plot_colors)] + '--')  plot_colors[i % len(plot_colors)] + 's'

    plt.xlabel("Generations") # , size=32
    plt.xticks((range(0,generations_to_include+1, max_generations//10))) # , size=32
    plt.ylabel("Permille(1/1000) correct") # , size=32
    plt.yticks((range(0, 1000 + 1, 100))) # , size=32
    plt.grid(linestyle='-.', linewidth=0.1)
    legend = ax1.legend(loc='upper left', shadow=True, prop={'size': 12}, ncol=5)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    # Save the full figure...
    file_location = os.path.dirname(os.path.realpath(__file__))
    #fig.savefig(file_location + name+".pdf", format="pdf", bbox_inches='tight')
    #fig.savefig(file_location + name+".png", dpi=1200, bbox_inches='tight')
    fig.savefig(file_location + name,dpi=600, bbox_inches='tight')
    i += 1

ea_runs_table = []
for i in range(1, len(ea_info_files_names)+1):

    ea_run = ea_runs.get(i)
    ind_info = best_ind_info.get(i)
    no_generations = len(ea_run.get("best_fitness"))
    all_rules = ind_info.get("full_size_rule_list")
    distinct_rules = []
    for rule in all_rules:
        if rule not in distinct_rules:
            distinct_rules.append(rule)

    #distinct_rules = "[" + tabulate.tabulate([distinct_rules], tablefmt="plain", stralign="left") + "]"
    distinct_rules = distinct_rules.__str__()
    ea_runs_table.append([i, no_generations, max(ea_run.get("best_fitness")), distinct_rules])

# Make table
print(tabulate.tabulate(ea_runs_table, tablefmt="latex", floatfmt=".2f"))
#print(" \\\\\n".join([" & ".join(map(str,line)) for line in [[123411,1],[2,2]]]))
