# api.py
# Provides the Python API for interacting with the hippocampus model

import requests
import time
import random
import numpy as np
import json

SERVER_URL="http://localhost"
PORT="1323"

class HipAPI:

    def __init__(self):
        # do nothing
        print("Creating API")

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
        # nothing happens if request was sucessful
        try:
            response = requests.request(verb, url=url, data=data)
            response.raise_for_status()
        except:
            return "ERROR: " + response.text
        else:
            return response.text
            #return "Request: " + verb + " " + str(data) + " to/from " + url + " was succesful."

    # update training data file to filename
    def UpdateTrainingData(self, filename):
        api_endpoint = "/dataset/train/update"
        data = {"filename":filename}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    # updates input pattern data file to filename
    def UpdateInputData(self, filename):
        api_endpoint = "/dataset/input/update"
        data = {"filename":filename}

        return self.MakeRequest('PUT', self.MakeURLString(api_endpoint), data)

    # tests a pattern
    def TestPattern(self, pattern):
        """
        Tests a pattern in the model.

        pattern: Numpy array representing a 2-D pattern. Numpy array MUST be integers.

        Returns: The pattern to which the recall is most similar and the Hamming distance between the recalled pattern and the most similar pattern.
        """

        api_endpoint = "/model/testpattern"

        d = {'shape': json.dumps(pattern.shape), 'pattern': json.dumps(np.ndarray.tolist(pattern))}

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), d)

    def StartTraining(self, parameters):
        """
        Starts model training from scratch.

        parameters: Dictionary of parameters to specify. If none, default settings will be used. Valid parameters are "maxruns" and "maxepcs".
        """

        api_endpoint = "/model/train"

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), parameters)


# Updates the training data to be drawn from wheedata.csv
# Filename is given as a relative path from where the model is 
# to where the dataset is.
api = HipAPI()

TEST_TESTITEM = False
TEST_STARTTRAINING = True

if TEST_STARTTRAINING:
    response = api.StartTraining({"maxruns":1, "maxepcs":50, "yeet":10})
    print(response)

# response = api.UpdateTrainingData("datasets/no_context/wheedata.csv")
# print(response)
# response = api.UpdateInputData("datasets/no_context/wheedata.csv")
# print(response)

if TEST_TESTITEM:
    # testAB's ab_0 pattern
    bitstring = '0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0'
    bitlist = bitstring.split(",") # convert to list
    bitlist = [int(x) for x in bitlist] # convert to ints
    #bitlist = bitlist[144:] # get just test pattern
    arr = np.asarray(bitlist, dtype="int") # convert to numpy array
    arr = np.reshape(arr, (6,2,3,4)) # reshape it to be the correct tensor shape

    response = api.TestPattern(arr)
    print(response)

    # a = np.zeros(10, dtype=int)
    # a[:5] = 1
    # np.random.shuffle(a)
    # a = a.reshape((2,5))
    # print(a)

