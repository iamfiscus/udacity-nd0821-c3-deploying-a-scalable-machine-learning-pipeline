"""
Script to post to FastAPI instance for model inference
author: iamfiscus
Date: 2023-03
"""

import requests
import json

url = "https://udacity-ml-devops.herokuapp.com/inference"


# Sample
sample = {
    'age': 38,
    'workclass': "Private",
    'fnlgt': 215646,
    'education': "HS-grad",
    'education_num': 9,
    'marital_status': "Divorced",
    'occupation': "Handlers-cleaners",
    'relationship': "Not-in-family",
    'race': "White",
    'sex': "Male",
    'capital_gain': 0,
    'capital_loss': 0,
    'hours_per_week': 40,
    'native_country': "United-States"
}

data = json.dumps(sample)

# Request to API
res = requests.post(url, data=data)

# Response
print("Status:", res.status_code)
print("Data:")
print(res.json())
