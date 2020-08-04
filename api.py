# Copyright (c) 2020, Stephen Polcyn. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# api.py
# Provides the Python API for interacting with the hippocampus model

import time
import random
import json
import logging

import requests
import numpy as np

import hip_util

import tensor_pb2
import dataset_update_pb2
import test_item_pb2

# basic default parameters for the model server
SERVER_URL="http://localhost"
PORT="1323"

class HipAPI:

    def __init__(self):
        # configure the module logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing API")

    def MakeURLString(self, api_endpoint):
        """
        Makes the full request URL for an API request from an endpoint URL.

        Args:
            api_endpoint (string): The path for the endpoint, starting with '/'

        Returns:
            string: The full request URL for the API endpoint.
        """

        return SERVER_URL + ":" + PORT + api_endpoint

    def MakeRequest(self, verb, url, data, headers=None):
        """
        Send an HTTP request using the provided parameters and format response into desired tuple.

        Args:
           verb (string): The HTTP verb to use (e.g., GET, POST, PUT)
           url (string): The full URL to send the HTTP request to, including the REST endpoint
           data (dict): The data to send in the body of the request as JSON.

        Returns:
            (requests.Response, bool): The full response to the HTTP request, and a bool indicating whether an error was returned by the server.
        """

        response = requests.request(verb, url=url, data=data, headers=headers)

        return response, response.ok

    def UpdateTrainingDataFile(self, filename):
        """
        Update the model's training data.

        Args:
            filename (string): Full path to the Emergent-formatted CSV file containing the training patterns. Must have both Input and ECout patterns.

        Returns:
            (requests.Response, bool): The full response to the HTTP request, and a bool indicating whether an error was returned by the server.
        """

        api_endpoint = "/dataset/train/update"
        data = {"source":"file", "filename":filename}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    def UpdateTrainingDataPatterns(self, patterns):
        """
        Update training data to the patterns provided.

        Args:
            patterns (list of 4-D Numpy arrays): The patterns to train the model on.

        Returns:
            (requests.Response, bool): The full response to the HTTP request, and a bool indicating whether an error was returned by the server.
        """

        api_endpoint = "/dataset/train/update"

        dataset = hip_util.make_tensor_from_numpy(np.asarray(patterns))

        update = dataset_update_pb2.DatasetUpdate()
        update.version = 1
        update.source = "body"
        update.dataset.CopyFrom(dataset)

        data = update.SerializeToString()

        return self.MakeRequest('PUT',
                                self.MakeURLString(api_endpoint),
                                data,
                                headers={'Content-Type':'application/octet-stream'})

    def TestPattern(self, corruptedPattern, targetPattern):
        """
        Tests a pattern in the model. Patterns must both be the same shape as each other and as the input layer of the hippocampus model.

        Args:
            corruptedPattern (4-D Numpy integer array): Original pattern corrupted in some way.
            targetPattern (4-D Numpy integer array): Complete original pattern with no data removed.

        Returns:
            (requests.Response, bool): The full response to the HTTP request, and a bool indicating whether an error was returned by the server. In particular, the response contains (in JSON format):
            1) The pattern (with all Emergent etensor properties, including shape, stride, dimension names, and actual pattern values) to which the recalled pattern is most similar
            2) The Hamming distance between the recalled pattern and the target pattern.
        """

        api_endpoint = "/model/testpattern"

        assert corruptedPattern.shape == targetPattern.shape

        test_item = test_item_pb2.TestItem()
        test_item.version = 1
        test_item.corrupted_pattern.CopyFrom(
                hip_util.make_tensor_from_numpy(corruptedPattern))
        test_item.target_pattern.CopyFrom(
                hip_util.make_tensor_from_numpy(targetPattern))

        data = test_item.SerializeToString()

        return self.MakeRequest('POST',
                                 self.MakeURLString(api_endpoint),
                                 data,
                                 headers={'Content-Type':'application/octet-stream'})

    def StartTraining(self, maxruns = 1, maxepcs = 50):
        """
        Starts model training from scratch. This method will wait until the training is complete, so it could take some time.

        Args:
            maxruns (int): Number of model runs to perform (independent times to retrain)
            maxecps (int): Number of epochs per run (epoch: train/test cycle with each item in test set)

        Returns:
            (requests.Response, bool): The full response to the HTTP request, and a bool indicating whether an error was returned by the server. In particular, the response contains (in JSON format):
            1) A string with information about the training time and parameters
        """
        self.logger.debug("Starting training with parameters maxruns: %i, maxepcs: %i", maxruns, maxepcs)

        # configure parameters dict
        parameters = {}
        parameters["maxruns"] = maxruns
        parameters["maxepcs"] = maxepcs

        api_endpoint = "/model/train"

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), parameters)

    def Step(self, cues, targets, iterations):
        """
        Steps the model forward one cycle. Similar to OpenAI gym's RL setup. Model must be trained prior to calling "step".
        Generally, this method should be re-implemented for an RL environment using the basic API methods.
        This method can serve as a useful template for creating such methods.

        Args:
            cues (Python list of 2-D Numpy arrays): Partial patterns to be tested.
            targets (Python list of 2-D Numpy arrays): Original patterns to be compared.
            iterations (int): Number of tests to perform on each pattern.

        Returns:
            Numpy array: The average reward for the step.
        """

        rewards = np.zeros((iterations, len(cues)))

        for i in range(iterations):
            for j, cue in enumerate(cues):
                response, success = self.TestPattern(cue, targets[j])
                distance = json.loads(response.text)["Distance"] # extract the distance from the response
                reward = np.size(cue) - distance # max reward is size of the pattern (when distance = 0) -- could normalize
                rewards[i][j] = reward

        return np.mean(rewards)

# -------------------------------------------------------------------------------- #
# A variety of sample API interactions that demonstrate simple functionality
# -------------------------------------------------------------------------------- #

TEST_TESTITEM = False
TEST_STEP = False
TEST_STARTTRAINING = False
TEST_UTP = False

if TEST_STARTTRAINING:
    tst_hipapi = HipAPI()
    response, success = tst_hipapi.StartTraining(maxruns = 1, maxepcs = 50)
    print("Test Start Training\n", response.text)

if TEST_TESTITEM:
    tti_hipapi = HipAPI()
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    arr = np.asarray(bitlist, dtype="float") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    response, success = tti_hipapi.TestPattern(arr, arr)
    print("Test Item\n", response.text)

if TEST_STEP:
    ts_hipapi = HipAPI()
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    reward = ts_hipapi.Step([arr], [arr], 5)
    print("Test Step\n", reward)

if TEST_UTP:
    utp_hipapi = HipAPI()
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    response, success = utp_hipapi.UpdateTrainingDataPatterns([arr, arr, arr, arr, arr, arr])
    print("Test UTP\n", response.request.body)
