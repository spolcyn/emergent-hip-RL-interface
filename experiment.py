# experiment.py
# Author: Stephen Polcyn
# Runs various experiments using the hippocampus API
import numpy as np
import api as hipapi
import logging
import time
from tqdm import tqdm

def MemoryVsPatternCount(minPatterns = 10, maxPatterns = 20, step = 1, trials = 10, trainingEpochs = 50):
    """
    Tests the performance of the model over increasing number of patterns.

    Args:
        maxPatterns: Maximum number of patterns to train on at once
        minPatterns: Minimum number of patterns to train on at once
        step: Number of patterns to increase by for each new trial
        trials: Number of times to test each item per condition (results averaged over trials for each item to give total per performance in given condition)

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
    sparsity = .75 # % of neurons that should be inactive

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
    results = np.zeros(numConditions) # store avg memory for each condition

    for c in tqdm(range(numConditions)):

        logging.info("Starting condition: %i / %i", c, numConditions)

        currentData = patternlist[:minPatterns + step * c]

        # send new patterns and train model
        response, success = ha.UpdateTrainingDataPatterns(currentData)
        if success:
            logging.debug("Successful: %s", response.text)
        else:
            logging.debug("Failure, %s, %s", response.status_code, response.text)

        response, success = ha.StartTraining(maxepcs=trainingEpochs)
        if success:
            logging.info("Training Successful: %s", response.text)
        else:
            logging.debug("Failure, %s", response.text)

        # run trials for condition c
        for t in range(trials):
            # run all patterns within trial
            successfulRecalls = 0

            for idx, p in enumerate(currentData): 
                response, success = ha.TestPattern(p)
                closestPattern = response.json()["ClosestPattern"]["Values"] # returned as list
                recallSuccessful = (closestPattern == p.reshape((totalValues)).tolist())
                
                if recallSuccessful:
                    successfulRecalls += 1

                #logging.info("Test %i successful", idx)
                #logging.info("Test %i unsuccessful", idx)
                #logging.debug("Closest pattern: %s", closestPattern)
                #logging.debug("Input pattern: %s", p.reshape((totalValues)).tolist())

            results[c] += (successfulRecalls / len(currentData)) # normalize accross patterns

        results[c] /= trials # normalize across trials

    # build results dict
    rd = {}
    for i in range(numConditions):
        rd[minPatterns + step * i] = results[i]

    return rd

# setup and run experiment
logging.basicConfig(level=logging.INFO)
start = time.monotonic()
results = MemoryVsPatternCount(minPatterns = 1000, maxPatterns = 1000, step = 1, trials = 1, trainingEpochs=1)

# report results
end = time.monotonic()
print("Finished in", end - start, "seconds")
print("Results:", results)

