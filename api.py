# api.py
# Provides the Python API for interacting with the hippocampus model

import requests
import time

SERVER_URL="http://localhost"
PORT="1323"

def MakeURLString(api_endpoint):
    """Makes the full request URL for an API request

    parameters:
    api_endpoint (string): The path for the endpoint, starting with '/'
    """

    return SERVER_URL + ":" + PORT + api_endpoint

# wrap error handling
def MakeRequest(verb, url, data):
    # make request and ensure response went through ok by raising an error if 
    # bad HTTP response code (4XX or 5XX)
    # nothing happens if request was sucessful
    try:
        response = requests.request(verb, url=url, data=data)
        response.raise_for_status()
    except:
        return "ERROR: " + response.text
    else:
        return "Success"
        #return "Request: " + verb + " " + str(data) + " to/from " + url + " was succesful."

# update training data file to filename
def UpdateTrainingData(filename):
    api_endpoint = "/dataset/train/update"
    data = {"filename":filename}

    return MakeRequest('PUT', MakeURLString(api_endpoint), data)

# updates input pattern data file to filename
def UpdateInputData(filename):
    api_endpoint = "/dataset/input/update"
    data = {"filename":filename}

    return MakeRequest('PUT', MakeURLString(api_endpoint), data)

# Updates the training data to be drawn from wheedata.csv
# Filename is given as a relative path from where the model is 
# to where the dataset is.
response = UpdateTrainingData("wheedata.csv")
print(response)
response = UpdateInputData("wheedata.csv")
print(response)

t = time.monotonic()
for i in range(1000):
    UpdateInputData("wheedata.csv")

print(time.monotonic() - t)
