import spacy
from fastapi import FastAPI
from typing_extensions import TypedDict

NLP = spacy.load("en_core_web_sm")


def get_entities(text):
    doc = NLP(text)
    entities = []
    for ent in doc.ents:
        entities.append(ent.text)
    return entities


app = FastAPI()


class Data(TypedDict):
    text: str


@app.get("/")
def index():
    return {"message": "This is NER api."}


@app.post("/score_ner")
def score_ner(data: Data):
    entities = get_entities(data["text"])
    return {"entities": entities}
