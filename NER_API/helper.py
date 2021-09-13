import requests

# Helper fucntions
def get_entities(text):
    ent_dict = requests.post("http://localhost:5000/score_ner", 
                             json={"text": text}).json()
    