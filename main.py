# Put the code for your API here.
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd

from ml.data import process_data
from ml.model import inference, load


"""
This function takes in a string as an argument and returns the same string with all underscores replaced with hyphens.
It uses the replace() method to do this.
Input: String
Output: String
"""


def hyphenazier(string: str) -> str:
    return string.replace('_', '-')


"""
This class is a subclass of BaseModel and is used to store census data.

It contains the following fields: age(int), workclass(str), fnlgt(int),
education(str), education_num(int), marital_status(str), occupation(str),
relationship(str), race(str), sex(str), capital_gain(int), capital_loss(int),
hours_per_week(int) and native_country(str).
"""


class CensusData(BaseModel):
    age: int = Field(example=38)
    workclass: str = Field(example='Private')
    fnlgt: int = Field(example=215646)
    education: str = Field(example='HS-grad')
    education_num: int = Field(example=9)
    marital_status: str = Field(example='Divorced')
    occupation: str = Field(example='Handlers-cleaners')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(example=0)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example='United-States')
    """
    The Config class allows for alias generation using hyphenazier
    and allows population by field name.
    """
    class Config:
        alias_generator = hyphenazier
        allow_population_by_field_name = True


"""
This code creates an API using FastAPI with the title
"iamfiscus Inference API",
a description of "An API that takes a sample and runs an inference",
and version 1.0.0.
"""
app = FastAPI(title="iamfiscus Inference API",
              description="An API that takes a sample and runs an inference",
              version="1.0.0")
"""
It also loads three models from the ./model directory:
random_forest.pkl, encoder.pkl, and lbinarizer.pkl.
"""
model = load('./model/random_forest.pkl')
encoder = load('./model/encoder.pkl')
lb = load('./model/lbinarizer.pkl')

# Finally, it creates a list of categorical features for use in the API.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


"""
This code creates a route "/" for an app using FastAPI.
When the route is accessed, the function greet_user() is executed,
which returns a dictionary containing the message "Hello World.
This app is a FastAPI for Project 3 of Udacity's ML for DevOps Nanodegree".
"""


@app.get("/")
async def greet_user():
    return {
        "data": "Hello World. This app is a FastAPI for Project 3 of Udacity's ML for DevOps Nanodegree"
    }

"""
This code is a post request for an inference endpoint. 
It takes in a CensusData object as an input, and converts it into a dictionary. 
It then uses the process_data function to process the data, and passes it into the inference function to get a list of results. 

The list of results is converted into a dictionary with each result being either '<=50k' or '>50k'. 

Finally, the dictionary is returned. 
"""


@app.post("/inference")
async def model_predict(endpoint_input: CensusData):
    endpoint_input_dict = endpoint_input.dict(by_alias=True)
    model_input = pd.DataFrame([endpoint_input_dict])

    processed_model_input, _, _, _ = process_data(
        model_input, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    inference_list = list(inference(model, processed_model_input))

    result = {}
    for i in range(len(inference_list)):
        if inference_list[i] == 0:
            result[i] = '<=50k'
        else:
            result[i] = '>50k'

    return {"result": result}


# This code checks if the current file is being run as the main program, and if so, it does nothing.
if __name__ == '__main__':
    pass
