# Named Entity Recognition + REST API

This code wraps NER model from spacy into a fastapi based REST api.

Start the server:
$ uvicorn main:app --reload --port 8000

Access api in the browser at following link:
http://127.0.0.1:8000/docs

Depdendencies:
- fastapi
- uvicorn
- spacy