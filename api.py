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
        """

        api_endpoint = "/model/testpattern"
        print(a.shape)
        d = {'shape': json.dumps(a.shape), 'pattern': json.dumps(np.ndarray.tolist(pattern))}

        return self.MakeRequest('POST', self.MakeURLString(api_endpoint), d)

# Updates the training data to be drawn from wheedata.csv
# Filename is given as a relative path from where the model is 
# to where the dataset is.
api = HipAPI()

response = api.UpdateTrainingData("wheedata.csv")
print(response)
response = api.UpdateInputData("wheedata.csv")
print(response)

a = np.zeros(10, dtype=int)
a[:5] = 1
np.random.shuffle(a)
a = a.reshape((2,5))
print(a)
response = api.TestPattern(a)
print(response)
