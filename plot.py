# Copyright (c) 2020, Stephen Polcyn. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# plot.py
# Plots the saved results of experiment.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot

results = np.load("results.npy")
minPatterns = 2; maxPatterns = 20; step = 2; corruptionRatios = np.linspace(0, 1, num=10, endpoint=False)

fig, ax = plt.subplots(1,1, figsize=(9,9))
img = ax.imshow(results, aspect='auto', cmap='Reds', interpolation='none')
fig.colorbar(img)

y_labels = [str(i) for i in np.arange(minPatterns, maxPatterns + 1, step)]
ax.set_yticks(range(int((maxPatterns-minPatterns)/step) + 1))
ax.set_yticklabels(y_labels, fontsize=12)

x_labels = np.around(corruptionRatios, 1)
ax.set_xticks(range(len(corruptionRatios)))
ax.set_xticklabels(x_labels, fontsize=12)

ax.set_ylabel("Number of Patterns", fontsize=18)
ax.set_xlabel("Corruption Ratio", fontsize=18)
ax.set_title("Effect of Pattern Count and Corruption Ratio\non Recall Accuracy (5 Training Epochs)", fontsize=20)
        

plt.show()
