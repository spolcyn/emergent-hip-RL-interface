# experiment.py
# Author: Stephen Polcyn
# Runs an experiment using the Emergent hippocampus Python API

import numpy as np
import api as hipapi
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import pyplot

# create module logger
logger = logging.getLogger(__name__)

def CorruptPattern(pattern, ratio):
    """
    Corrupts a pattern for testing.

    Args:
        pattern (4-D Numpy Array): The pattern to corrupt.
        ratio (float): The amount of information to remove, as a decimal in [0, 1].

    Returns:
        Numpy array: A copy of the original pattern, corrupted the prescribed amount.
    """


    # calc number of units
    shape = list(pattern.shape)
    assert len(shape) == 4 # ensure its a 4-d array
    slots = shape[0] * shape[1]

    corrupted = pattern.copy() # copy to avoid modifying source

    neuronsToCorrupt = int(np.sum(corrupted) * ratio)
    corruptedNeurons = 0

    logger.debug("Corrupting pattern with ratio %f, total neurons to corrupt: %i", ratio, neuronsToCorrupt)

    # iterate through the pattern, switching the first neuronsToCorrupt
    # active neurons to inactive
    for x in np.nditer(corrupted, op_flags = ['readwrite']):
        logger.debug("Corrupted Neurons: %i, Neurons To Corrupt Total: %i", corruptedNeurons, neuronsToCorrupt)

        if (corruptedNeurons == neuronsToCorrupt):
            break

        # if not complete, try to corrupt one
        if x[...] == 1:
            x[...] = 0
            corruptedNeurons += 1

    return corrupted

def MemoryVsPatternCount(minPatterns = 10, maxPatterns = 20, step = 1, trials = 10, trainingEpochs = 50, corruptionRatios = [.5], sparsity = 0.75):
    """
    Tests the performance of the model over increasing number of patterns.

    Args:
        maxPatterns (int): Maximum number of patterns to train on at once.
        minPatterns (int): Minimum number of patterns to train on at once.
        step (int): Number of patterns to increase by for each new trial.
        trials (int): Number of times to test each item per condition (results averaged over trials for each item to give total per performance in given condition).
        trainingEpochs (int): Number of epochs to train the model for when training on each dataset.
        corruptionRatios (Numpy array of floats): Amount of the image to corrupt. 0 means no corruption, 1 means total deactivation of every neuron in the pattern. Values must be in [0, 1].
        sparsity (float): Percentage of neurons as a decimal that should be inactive in the generated patterns.

    Returns: 
        Numpy array: Array contains percentage of items correctly recalled for each trial size.
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

    # create all patterns
    for i in range(maxPatterns):

        # configure pattern
        pat = np.ones(totalValues, dtype=int) 
        pat[:int(totalValues*sparsity)] = 0
        np.random.shuffle(pat)
        pat = np.reshape(pat, tuple(shape))

        # add to list
        patternlist.append(pat)

    # run the experiment
    numConditions = int((maxPatterns-minPatterns)/step) + 1 # number of different datasets to use
    results = np.zeros((numConditions, len(corruptionRatios))) # store avg memory for each condition and corruption ratio

    for c in tqdm(range(numConditions)):

        logger.info("Starting condition: %i / %i", c, numConditions)
        start = time.monotonic()

        # slice the amount of patterns currently being used
        currentData = patternlist[:minPatterns + step * c]
        logger.debug("CurrentData: %s", currentData)

        # send new patterns 
        response, success = ha.UpdateTrainingDataPatterns(currentData)
        if success:
            logger.debug("Successful: %s", response.text)
        else:
            logger.debug("Failure, %s, %s", response.status_code, response.text)

        # train model
        response, success = ha.StartTraining(maxepcs=trainingEpochs)
        if success:
            logger.debug("Training Successful: %s", response.text)
        else:
            logger.debug("Failure, %s", response.text)

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

                    logger.debug("Closest pattern: %s", closestPattern)
                    logger.debug("Input pattern: %s", p.reshape((totalValues)).tolist())
                    logger.debug("Corrupted pattern: %s", corrupted.reshape((totalValues)).tolist())

                results[c][r_idx] += (successfulRecalls / len(currentData)) # normalize accross patterns

        results[c] /= trials # normalize across trials

        totalTime = time.monotonic() - start
        logger.info("Completed condition %i with %i patterns, average accuracies %s, total time: %f", c, len(currentData), results[c], totalTime)

    return results

"""
# setup logging
logging.basicConfig()
logger.info("Starting")

# setup parameters
minPatterns = 2
maxPatterns = 20
step = 2
corruptionRatios = np.linspace(0, 1, num=10, endpoint=False)
sparsity = .75

# time and run experiment
start = time.monotonic()
results = MemoryVsPatternCount(minPatterns = minPatterns, maxPatterns = maxPatterns, step = step, trainingEpochs=5, corruptionRatios=corruptionRatios, sparsity = .75)
end = time.monotonic()

# report results
print("Finished in", end - start, "seconds")
print("Results:", results)

# save results for use with plot.py
np.save("results", results)
"""
