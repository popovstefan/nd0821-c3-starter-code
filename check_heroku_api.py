"""
POSTs to the Heroku app that is supposed to be live.

author Stefan Popov
date Apr 2022
"""
import requests

data = {
    "age": 45,
    "workclass": "State-gov",
    "education": "HS-grad",
    "maritalStatus": "Divorced",
    "occupation": "Protective-serv",
    "relationship": "Unmarried",
    "race": "Black",
    "sex": "Female",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}
response = requests.post('https://udacity-c3-app.herokuapp.com/inference', json=data)
print("Response Code", response.status_code)
print("Response Body", response.json())
assert response.status_code == 200
