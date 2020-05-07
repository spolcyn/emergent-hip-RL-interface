# experiment.py
# Author: Stephen Polcyn
# Runs various experiments using the hippocampus API
import numpy as np
import api as hipapi
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import pyplot

logging.basicConfig(level=logging.INFO)

def CorruptPattern(pattern, ratio):
    """
    Corrupts a pattern for testing.

    Args:
        pattern: The pattern to corrupt
        ratio: The amount of information to remove, as a decimal in [0, 1]
    """

    # calc number of units
    shape = list(pattern.shape)
    assert len(shape) == 4 # ensure its a 4-d array
    slots = shape[0] * shape[1]

    corrupted = pattern.copy() # copy to avoid modifying source


    corruptUnits = np.random.choice(slots, int(ratio * slots), replace=False)
    logging.debug("Units to corrupt: %s", str(corruptUnits))

    for u in corruptUnits:
        row = int(u/shape[1])
        column = u % shape[1]
        logging.debug("Row: %i", row)
        logging.debug("Column: %i", column)
        corrupted[row][column] = np.zeros((shape[2], shape[3]))

    return corrupted

def MemoryVsPatternCount(minPatterns = 10, maxPatterns = 20, step = 1, trials = 10, trainingEpochs = 50, corruptionRatios=[.5]):
    """
    Tests the performance of the model over increasing number of patterns.

    Args:
        maxPatterns: Maximum number of patterns to train on at once
        minPatterns: Minimum number of patterns to train on at once
        step: Number of patterns to increase by for each new trial
        trials: Number of times to test each item per condition (results averaged over trials for each item to give total per performance in given condition)
        corruptionRatios: Amount of the image to corrupt. 0 means no corruption, 1 means total deactivation of every neuron in the pattern.

    Returns: 
        numpy array with % items correctly recalled for each trial size
    """
    
    ha = hipapi.HipAPI()

    # ensure valid range specified
    if (maxPatterns - minPatterns) % step != 0:
        print("maxPattern is not minPattern + step * (some integer)")
        return
    if minPatterns < 1:
        print("minPatterns is too small (value:", minPatterns, ")")
        return

    patternlist = []

    # config pattern shape-related properties
    shape = [6,2,3,4] 
    totalValues = 1
    for v in shape:
        totalValues *= v
    sparsity = .75 # % of neurons that should be inactive in generated patterns

    # create all patterns
    for i in range(maxPatterns):

        # create the pattern
        pat = np.ones(totalValues, dtype=int) 
        pat[:int(totalValues*sparsity)] = 0
        np.random.shuffle(pat)
        pat = np.reshape(pat, tuple(shape))

        patternlist.append(pat)

    # run the experiment
    numConditions = int((maxPatterns-minPatterns)/step) + 1 # number of different datasets to use
    results = np.zeros((numConditions, len(corruptionRatios))) # store avg memory for each condition and corruption ratio

    for c in range(numConditions):

        logging.info("Starting condition: %i / %i", c, numConditions)
        start = time.monotonic()

        currentData = patternlist[:minPatterns + step * c]

        # send new patterns and train model
        response, success = ha.UpdateTrainingDataPatterns(currentData)
        if success:
            logging.debug("Successful: %s", response.text)
        else:
            logging.debug("Failure, %s, %s", response.status_code, response.text)

        response, success = ha.StartTraining(maxepcs=trainingEpochs)
        if success:
            logging.debug("Training Successful: %s", response.text)
        else:
            logging.debug("Failure, %s", response.text)

        # run different cue corruption ratios for condition c
        for r_idx, r in enumerate(corruptionRatios):
            # run trials for condition c and cue corruption ratio r
            for t in range(trials):
                # run all patterns within trial
                successfulRecalls = 0

                for idx, p in enumerate(currentData): 
                    corrupted = CorruptPattern(p, r)
                    response, success = ha.TestPattern(corrupted, p)

                    closestPattern = response.json()["ClosestPattern"]["Values"] # returned as list
                    recallSuccessful = (closestPattern == p.reshape((totalValues)).tolist()) # test if the closest pattern is the same as the target
                    
                    if recallSuccessful:
                        successfulRecalls += 1

                    logging.debug("Closest pattern: %s", closestPattern)
                    logging.debug("Input pattern: %s", p.reshape((totalValues)).tolist())
                    logging.debug("Corrupted pattern: %s", corrupted.reshape((totalValues)).tolist())

                results[c][r_idx] += (successfulRecalls / len(currentData)) # normalize accross patterns

        results[c] /= trials # normalize across trials
        totalTime = time.monotonic() - start
        logging.info("Completed condition %i with %i patterns, average accuracies %s, total time: %f", c, len(currentData), results[c], totalTime)

    # build results dict
    """
    rd = {}
    for i in range(numConditions):
        rd[minPatterns + step * i] = results[i]
    """

    return results

# setup and run experiment
logging.basicConfig()
logging.info("Starting")
start = time.monotonic()
minPatterns = 2; maxPatterns = 20; step = 2; corruptionRatios = np.linspace(0, 1, num=10, endpoint=False)
results = MemoryVsPatternCount(minPatterns = minPatterns, maxPatterns = maxPatterns, step = step, trials = 10, trainingEpochs=1, corruptionRatios=corruptionRatios)

# report results
end = time.monotonic()
print("Finished in", end - start, "seconds")
print("Results:", results)

# plot
"""
lists = sorted(d.items())
x, y = zip(*lists)
plt.plot(x,y)
plt.show()
"""
"""
arr = np.arange(100)
arr = arr.reshape((10,10))
arr = arr / 100

results = arr
"""

fig, ax = plt.subplots(1,1, figsize=(9,8))
print(results.shape)
img = ax.imshow(results, aspect='auto', cmap='Reds', interpolation='none')
fig.colorbar(img)
#ax.set_ylim(bottom = 0, top = results.shape[0])

y_labels = [str(i) for i in np.arange(minPatterns, maxPatterns + 1, step)]
ax.set_yticks(range(int((maxPatterns-minPatterns)/step) + 1))
ax.set_yticklabels(y_labels)

x_labels = np.around(corruptionRatios, 1)
ax.set_xticks(range(len(corruptionRatios)))
ax.set_xticklabels(x_labels)

ax.set_ylabel("Number of Patterns")
ax.set_xlabel("Corruption Ratio")
ax.set_title("Recall Accuracy as Function of Number of Patterns and Corruption Ratio")
        

plt.show()
