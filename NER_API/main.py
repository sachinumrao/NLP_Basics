import spacy
from fastapi import FastAPI

NLP = spacy.load("en_core_web_sm")
def get_entities(text):
    doc = NLP(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities

app = FastAPI()

@app.get("/")
def index():
    return {"message": "This is NER api."}

@app.post("/score_ner")
def score_ner(text: str):
    entities = get_entities(text)
    return {"entities": entities}