import pickle
import pandas as pd
import requests
import json
from urllib.parse import urlencode
import re
from tqdm import tqdm
import copy
from rdflib import Graph
from SPARQLWrapper import SPARQLWrapper, JSON, N3
from pprint import pprint
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy
import os
from sentence_transformers import SentenceTransformer
import argparse


data = []
month = [12, 12, 12, 12, 12, 12 , 12]
year = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
list_file = []
for y_idx, y in enumerate(year):
    for m in range(1, month[y_idx]+1):
        if m < 10:
            list_file.append(str(y) + "-0" + str(m))
        else:
            list_file.append(str(y) + "-" + str(m))
for file in list_file:
    filename = "../data/pre_training/reddit/data/pretrain_data/pretrain_reddit_dbpedia_" + file + ".json"
    print("processing: " + filename)
    if os.path.isfile(filename):
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        f.close()
    else:
        print(filename + "does not exist.")


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def _calculate_similarity_with_senbert(response, list_candidates, threshold):
    list_candidates_text = [elem["subject"] + " " + elem["predicate"] + " " + elem["object"] for elem in list_candidates]
    sentences = [response] + list_candidates_text
    # sentence_embeddings
    sentence_embeddings = model.encode(sentences)
    scores = [float(cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[i].reshape(1, -1))[0][0]) for i in range(1, len(sentence_embeddings))]
    ids = []
    # print("response: " + response)
    # print("first predicate: " + list_candidates_text[0])
    # print("score: " + str(scores[0]))
    for idx, score in enumerate(scores):
        if score > threshold:
            ids.append(idx)
    ret_candidates = []
    for i in ids:
        ret_candidate = list_candidates[i]
        ret_candidate["sim_score"] = scores[i]
        ret_candidates.append(ret_candidate)
    # print("response: " + response)
    # print(ret_candidates)
    # print(len(ret_candidates))
    # print(scores[0])
    return ret_candidates

# count data points
print("Before filtering, it has " + str(len(data)) + " data points")
        
# Filtering bad data points
l_similarity_scores = []
filtered_data = []
for line in tqdm(data):
    context = line["context"]
    entities = line["entities"]
    response = line["response"]
    
    filtered_entities = _calculate_similarity_with_senbert(response["message"], entities, 0.35)
    if filtered_entities:
        filtered_data.append({"context": copy.deepcopy(context), "entities": copy.deepcopy(filtered_entities),"response": copy.deepcopy(response)})


output_pretrain_file_name = "../data/pre_training/reddit/data/pretrain_data/pretrain_reddit_dbpedia" + str(list_file[0]) + "-" + str(list_file[-1]) + ".json"
with open(output_pretrain_file_name, 'w') as fp:
    for elem in filtered_data:
        json.dump(elem, fp)
        fp.write('\n')
print("After filtering, it has " + str(len(filtered_data)) + " data points")
print(filtered_data[:2])