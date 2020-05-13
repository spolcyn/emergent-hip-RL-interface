# api.py
# Author: Stephen Polcyn
# Provides the Python API for interacting with the hippocampus model

import requests
import time
import random
import numpy as np
import json
import logging

logging.basicConfig(level=logging.DEBUG)

SERVER_URL="http://localhost"
PORT="1323"

class HipAPI:

    def __init__(self):
        logging.info("Creating API")

    def MakeURLString(self, api_endpoint):
        """Makes the full request URL for an API request

        parameters:
        api_endpoint (string): The path for the endpoint, starting with '/'
        """

        return SERVER_URL + ":" + PORT + api_endpoint

    # wrap error handling
    def MakeRequest(self, verb, url, data):
        # make request and ensure response went through ok by raising an error if 
        # bad HTTP response code (4XX or 5XX)
        # nothing happens if request was successful

        response = requests.request(verb, url=url, data=data)

        return response, (response.status_code < 400)

    # update training data file to filename
    def UpdateTrainingDataFile(self, filename):
        api_endpoint = "/dataset/train/update"
        data = {"source":"file", "filename":filename}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    # update training data to the patterns provided
    def UpdateTrainingDataPatterns(self, patterns):
        """
        Update training data to the patterns provided.

        Args:
            patterns: python list of 4-D numpy array patterns
        """
        api_endpoint = "/dataset/train/update"

        # format the list of numpy arrays into a format parseable by the backend
        jsonpats = [json.dumps(np.ndarray.tolist(p)) for p in patterns]
        jsonpats = [p.replace('[', '(') for p in jsonpats]
        jsonpats = [p.replace(']', ')') for p in jsonpats]

        # configure request body
        data = {"source":"body", "patterns":jsonpats, "shape": json.dumps(patterns[0].shape)}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    # updates input pattern data file to filename
    def UpdateInputData(self, filename):
        api_endpoint = "/dataset/input/update"
        data = {"filename":filename}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    # tests a pattern
    def TestPattern(self, corruptedPattern, targetPattern):
        """
        Tests a pattern in the model.

        corruptedPattern: Numpy array representing a 2-D pattern with data removed. Numpy array MUST be integers.
        targetPattern: Numpy array representing a 2-D pattern with all original data. Numpy array MUST be integers.

        Returns: The pattern (with all emergent etensor properties, including shape, stride, dimension names, and values) to which the recall is most similar and the Hamming distance between the recalled pattern and the target pattern.
        """

        api_endpoint = "/model/testpattern"

        assert corruptedPattern.shape == targetPattern.shape

        d = {'shape': json.dumps(corruptedPattern.shape), 'corruptedPattern': json.dumps(np.ndarray.tolist(corruptedPattern)), 'targetPattern':json.dumps(np.ndarray.tolist(targetPattern))}

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), d)

    def StartTraining(self, maxruns = 1, maxepcs = 50):
        """
        Starts model training from scratch.

        Args:
            maxruns: Number of model runs to perform (independent times to retrain)
            maxecps: Number of epochs per run (epoch: train/test cycle with each item in test set)
        """
        logging.debug("Starting training with parameters maxruns: %i, maxepcs: %i", maxruns, maxepcs)

        # configure parameters
        parameters = {}
        parameters["maxruns"] = maxruns
        parameters["maxepcs"] = maxepcs

        api_endpoint = "/model/train"

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), parameters)

    def Step(self, cues, iterations):
        """
        Steps the model forward one cycle. Similar to OpenAI gym's RL setup. Model must be trained prior to calling "step".
        Args:
            cues: python list of 2-d patterns to be tested
            iterations: number of tests to perform on each pattern

        Returns:
            observation (The pattern recalled) and reward (based on recall accuracy)
        """

        rewards = np.zeros((iterations, len(cues)))

        for i in range(iterations):
            for j, cue in enumerate(cues):
                response = self.TestPattern(cue)
                distance = json.loads(response)["Distance"]
                reward = np.size(cue) - distance # max reward is size of the pattern (when distance = 0) -- could normalize
                rewards[i][j] = reward

        return np.mean(rewards)

# Updates the training data to be drawn from wheedata.csv
# Filename is given as a relative path from where the model is 
# to where the dataset is.

TEST_TESTITEM = False
TEST_STEP = False
TEST_STARTTRAINING = False
TEST_UTP = False

if TEST_STARTTRAINING:
    response, success = hipapi.StartTraining(maxruns = 1, maxepcs = 50)
    print(response.text)

# response = hipapi.UpdateTrainingData("datasets/no_context/wheedata.csv")
# print(response)
# response = hipapi.UpdateInputData("datasets/no_context/wheedata.csv")
# print(response)

if TEST_TESTITEM:
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    #bitlist = bitlist[144:] # get just test pattern
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    response, success = hipapi.TestPattern(arr)
    print(response.text)

if TEST_STEP:
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    #bitlist = bitlist[144:] # get just test pattern
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    reward = hipapi.Step([arr], 5)
    print(reward)

if TEST_UTP:
    hipapi = HipAPI()
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    #bitlist = bitlist[144:] # get just test pattern
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    response, success = hipapi.UpdateTrainingDataPatterns([arr, arr, arr, arr, arr, arr])
    print(response.request.body)

    # a = np.zeros(10, dtype=int)
    # a[:5] = 1
    # np.random.shuffle(a)
    # a = a.reshape((2,5))
    # print(a)

