import requests
import streamlit as st
from helper import get_entities

# Set page title
st.title("Named Entity Recognition")

# Create a text input widget
text_input = st.text_area("Enter text here", "")

# Create a button widget
submit_btn = st.button("Submit")

# if submit call a NER api and run the text through the NER api
if submit_btn:
    ents = get_entities(text_input)

    if len(ents) > 0:
        st.write("Entities:")
        for ent in ents:
            st.write(ent)
    else:
        st.write("No entities found")
