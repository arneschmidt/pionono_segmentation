"""
This is a script to plot the agreement of the raters in terms of Cohens Kappa.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

matplotlib.rcParams.update({'font.size': 11})
# r1_ag = [0.36, 0.41, 0.48, 0.45, 0.48, 0.592, 0.6]
# r2_ag = [0.36, 0.46, 0.53, 0.53, 0.64, 0.369, 0.26]
# r3_ag = [0.41, 0.46, 0.69, 0.72, 0.60, 0.666, 0.642]
# r4_ag = [0.48, 0.53, 0.69, 0.72, 0.60, 0.673, 0.681]
# r5_ag = [0.45, 0.53, 0.72, 0.67, 0.58, 0.724, 0.741]
# r6_ag = [0.48, 0.64, 0.60, 0.58, 0.65, 0.629, 0.627]


# In each row: agreement to each other rater, 0.0 (later filled with mean), Pionono agreement
r1_ag = [0.36, 0.41, 0.48, 0.45, 0.48, 0.0,  0.6]
r2_ag = [0.36, 0.46, 0.53, 0.53, 0.64, 0.0, 0.26]
r3_ag = [0.41, 0.46, 0.69, 0.72, 0.60, 0.0, 0.642]
r4_ag = [0.48, 0.53, 0.69, 0.67, 0.58, 0.0,  0.681]
r5_ag = [0.45, 0.53, 0.72, 0.67, 0.65, 0.0, 0.741]
r6_ag = [0.48, 0.64, 0.60, 0.58, 0.65, 0.0, 0.627]

all_r = np.array([r1_ag, r2_ag, r3_ag, r4_ag, r5_ag, r6_ag])

fig, ax = plt.subplots(figsize=(4, 4.2))

labels = ['r=1', 'r=2', 'r=3', 'r=4', 'r=5', 'r=6', 'r-mean', 'Pionono']
colors = ['tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:green', 'k', 'k']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o', '*']
markersizes = [10,10,10,10,10,10,14,14]
indices = range(8)

for i in range(all_r.shape[0]):
    x = all_r[i]
    x[5] = np.mean(x[0:5])
    label_indices = [y for k, y in enumerate(indices) if k!=i]

    for j in range(x.shape[0]):
        idx = label_indices[j]

        ax.plot(x[j], i + 1, color=colors[idx], marker=markers[idx], markersize=markersizes[idx],
                linestyle='None', alpha=0.8)


legend_list = []
for i in [5,4,3,2,1,0]:
    legend_list.append(mlines.Line2D([], [], color=colors[i], marker=markers[i], linestyle='None',
                          markersize=markersizes[i], label=labels[i]))

for i in [6,7]:
    legend_list.append(mlines.Line2D([], [], color=colors[i], marker=markers[i], linestyle='None',
                          markersize=markersizes[i], label=labels[i]))

plt.legend(handles=legend_list)
plt.ylabel('Rater')
plt.xlabel("Unweighted Cohen\'s Kappa")
ax.grid(axis='y')
plt.tight_layout()
# ax.legend(numpoints=1)
# plt.legend()
plt.savefig("agreement_plot.png")

