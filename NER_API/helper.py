import requests
from pydantic import BaseModel

URL = "http://localhost:8000/score_ner"
HEADERS = {"Content-Type": "application/json"}

# Helper fucntions
def get_entities(text=None):
    if text is None:
        text = "Novak Djokovick lost the US Open 2021."
    payload = {"text": text}
    response = requests.post(URL, data=payload, headers=HEADERS)
    print(response.json())


if __name__ == "__main__":
    get_entities()
