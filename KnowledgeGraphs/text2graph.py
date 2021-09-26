import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import itertools
import time
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network 

nlp = spacy.load("en_core_web_sm")

def get_prop_nouns(sent):
    doc = nlp(sent)
    pos = [tok.i for tok in doc if tok.pos_ == "PROPN"]
    consecutives = []
    current = []
    for elt in pos:
        if len(current) == 0:
            current.append(elt)
        else:
            if current[-1] == elt - 1:
                current.append(elt)
            else:
                consecutives.append(current)
                current = [elt]
    if len(current) != 0:
        consecutives.append(current)
    tokens = [doc[consecutive[0]:consecutive[-1]+1].text for 
              consecutive in consecutives]
    
    tokens = [tok.lower() for tok in tokens]
    return tokens

def get_source_target_tuples(token_list):
    words = sorted(list(set(token_list)))
    combo = itertools.combinations(words, 2)
    return list(combo)

def get_sentences(text):
    sents = sent_tokenize(text)
    return sents

def make_graph(source_target_vals):
    df = pd.DataFrame(source_target_vals, columns=["Source", "Target"])
    df.groupby(df.columns.tolist(),as_index=False).size()
    print("graph size: ", len(df))
    data = nx.from_pandas_edgelist(df, source="Source", 
                                   target="Target",
                                   )
    

    net = Network(height='900px', width='100%')
    net.from_nx(data)
    return net


if __name__ == "__main__":
    graph_data = []
    fname = Path.home() / 'Data' / 'LM' / 'harry.txt'
    
    with open(fname, "r") as f:
        txt = f.read()
    
    sents = get_sentences(txt)
    
    for sent in tqdm(sents):
        prop_nouns = get_prop_nouns(sent)
        st_tups = get_source_target_tuples(prop_nouns)
        graph_data += st_tups
        
    graph = make_graph(graph_data)
    graph.hrepulsion()
    # save graph
    graph.show("Harry_Potter.html")
    
    