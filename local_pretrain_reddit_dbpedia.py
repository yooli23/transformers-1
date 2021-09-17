#!/usr/bin/env python
# coding: utf-8

# In[15]:

import os.path
import os
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
from multiprocessing import Pool
from sentence_transformers import SentenceTransformer
import numpy
import argparse


# In[16]:


def _rerank_info_with_tfidf(context, list_candidates, n_keep):
    list_candidates_text = [elem["subject"] + " " + elem["predicate"] + " " + elem["object"] for elem in list_candidates]
    train_set = [context] + list_candidates_text
    stopWords = stopwords.words('english')
    tfidf_vectorizer = TfidfVectorizer(stop_words = stopWords)
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)  #finds the tfidf score with normalization
    nparr_cos_sim = cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)  #here the first element of tfidf_matrix_train is matched with other three elements
    ids = list(nparr_cos_sim[0][1:].argsort()[::-1])
    if n_keep < len(ids):
        ids = ids[:n_keep]
    ret_candidates = [list_candidates[i] for i in ids]
    return ret_candidates

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def _rerank_info_with_senbert(response, list_candidates, n_keep):
    list_candidates_text = [elem["subject"] + " " + elem["predicate"] + " " + elem["object"] for elem in list_candidates]
    sentences = [response] + list_candidates_text
    # sentence_embeddings
    sentence_embeddings = model.encode(sentences)
    scores = [cosine_similarity(sentence_embeddings[0].reshape(1, -1), sentence_embeddings[i].reshape(1, -1))[0][0] for i in range(1, len(sentence_embeddings))]
    ids = list(numpy.argsort(scores)[::-1])
    if n_keep < len(ids):
        ids = ids[:n_keep]
    ret_candidates = [list_candidates[i] for i in ids]
    return ret_candidates

# In[18]:


profanity_regex = "|".join([
    r"\b(sex|porn|horny|pornographic|stripper)\b",
    r"\b(bitch|fuck)",
    r"\b(fart|shit)\b",
    r"\b(masturbate|testicles|penis|cock|testicles|(?<!moby )dick( ?!van dyke)|vaginas|booty hole|ass|vagina(|s)|virgin|butthole|pussy|ass|slut)\b",
    r"\b(boobs|boob)\b",
    r"\b(marijuana|cocaine)",
])
def _profanity_check(text):
    if re.search(profanity_regex, text):
        return True
    else:
        return False
    


# In[19]:


url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
def _url_check(text):
    if re.search(url_regex, text):
        return True
    else:
        return False
# _url_check("til that there were more than 9 actors in the fellowship . http://www.imdb.com/title/tt0120737/")


# In[20]:


def _clean_text(text):
    for prefix in ["til that ", "til : ", "til "]:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text
# _clean_text("til that in 1959 a greek politician called xenophon zolotas")


# In[21]:


# fact_filename = "/mnt/output/pre_training/reddit/data/2011-01.facts.txt"
# fact_df = pd.read_csv(fact_filename, sep="\t", header=None)
# fact_df.columns = ["hash_value", "subreddit_name", "conv_id", "domain_name", "fact"]
# # fact_df = fact_df.sort_values(by=["conv_id", "dialog_turn_num", "response_score"], ascending=[True, True, False])
# fact_df = fact_df.reset_index(drop=True)
# print(len(fact_df))
# fact_df = fact_df[fact_df['fact'].str.contains("<title>") & (fact_df['domain_name'] == "en.wikipedia.org")].drop_duplicates(subset=['conv_id']).reset_index(drop=True)
# print(len(fact_df))
# fact_df[:20]


# In[22]:


def _get_context(context):
    list_context = []
    list_context = context.split(' EOS ')
    list_context = [{"message": _clean_text(t), "role": "user" if idx % 2 == 0 else "agent"} for idx, t in enumerate(list_context[1:])]
    return list_context


# In[23]:
PREDICATE_MAPPING = {
    "Link from a Wikipage to another Wikipage": "is related to",
    "Wikipage redirect": "is related to",
}

def _mapping_predicate(predicate):
    if predicate in PREDICATE_MAPPING:
        return PREDICATE_MAPPING[predicate]
    return predicate

sparql = SPARQLWrapper('https://dbpedia.org/sparql')

def _get_entities_of_uri(dialog_context, response, wiki_URI):
    ret_list_triples = []
    QUERY_TEMPLATE = '''
    SELECT DISTINCT ?subject ?predicate ?object
    WHERE
    {
        { 
            OPTIONAL 
            {
                dbr:<URI>  ?p ?o .
                dbr:<URI> rdfs:label ?subject .
                ?p rdfs:label ?predicate .
                ?o rdfs:label ?object .
                FILTER langMatches(lang(?object),'en') .
                FILTER langMatches(lang(?predicate),'en') .
                FILTER langMatches(lang(?subject),'en') .
                FILTER(regex(str(?o), "http://dbpedia.org/resource", "i" )) .
                FILTER(regex(str(?p), "http://dbpedia.org/ontology/", "i" )) .
            } 
        }
        UNION
        { 
            OPTIONAL 
            { 
                ?s ?p dbr:<URI> .
                dbr:<URI> rdfs:label ?object .
                ?p rdfs:label ?predicate .
                ?s rdfs:label ?subject .
                FILTER langMatches(lang(?object),'en') .
                FILTER langMatches(lang(?predicate),'en') .
                FILTER langMatches(lang(?subject),'en') .
                FILTER(regex(str(?s), "http://dbpedia.org/resource", "i" )) .
                FILTER(regex(str(?p), "http://dbpedia.org/ontology/", "i" )) .
            } 
        }
    }
    ORDER BY ?p
    LIMIT 500
    '''
    if "(" in wiki_URI or ")" in wiki_URI or "," in wiki_URI or "'" in wiki_URI:
        wiki_URI = wiki_URI.replace("(", "\(").replace(")", "\)").replace(",", "\,").replace("'", "\\'")
    sparql.setQuery(QUERY_TEMPLATE.replace("<URI>", wiki_URI))
    sparql.setReturnFormat(JSON)
    try:
        qres = sparql.query().convert()
    except:
        print("sparql query error!" + "URI" + wiki_URI)
        return ret_list_triples
    for result in qres['results']['bindings']:
        if 'object' in result and 'value' in result['object'] and result['object']['value'] and 'value' in result['predicate'] and result['predicate']['value'] and 'value' in result['subject'] and result['subject']['value']:
            ret_list_triples.append({"subject": result['subject']['value'], "predicate": _mapping_predicate(result['predicate']['value']), "object": result['object']['value']})
    if ret_list_triples:
        str_input_context = ""
        for dict_message in dialog_context:
            str_input_context += (dict_message["role"] + ": " + dict_message["message"] + " ")
        str_input_context += (response["role"] + ": " + response["message"])
        ret_list_triples = _rerank_info_with_tfidf(str_input_context, ret_list_triples, 50)
        ret_list_triples = _rerank_info_with_senbert(response["message"], ret_list_triples, 50)
    return ret_list_triples


# In[24]:


def _get_uri_From_url(url):
    ret_uri = ""
    last_name = ""
    if url:
        last_name = url.rsplit('/', 1)[-1]
    if last_name:
        ret_uri = last_name.split('#')[0]
    return ret_uri

def _get_wiki_uri_name(pkl_data, conv_id):
    ret_uri_name = ""
    if conv_id in pkl_data and pkl_data[conv_id]['domain'] == "en.wikipedia.org":
        url = pkl_data[conv_id]['url']
        ret_uri_name = _get_uri_From_url(url)
    return ret_uri_name

def process_dataset(month_file):
    pre_train_file = "../data/pre_training/reddit/data/pretrain_data/pretrain_reddit_dbpedia_" + month_file + ".json"
    config_file = "../data/pre_training/reddit/data/pretrain_data/config" + month_file + ".json"
    # pre_train_file = "../data/pre_training/reddit/data/pretrain_data/test/pretrain_reddit_dbpedia_" + month_file + ".json"
    # config_file = "../data/pre_training/reddit/data/pretrain_data/test/config" + month_file + ".json"
    start_index = 0
    if os.path.isfile(config_file):
        with open(config_file) as config_json:
            config = json.load(config_json)
            start_index = config['start_index']
            print("start from: " + str(start_index) + " line!!!")

    pkl_data = {}
    pkl_file = "../data/pre_training/reddit/data/" + month_file + ".pkl"
    # pkl_file = "../data/pre_training/reddit/data/" + month_file + ".pkl"
    if not os.path.isfile(pkl_file):
        print(month_file + " pkl file not exist, exit!")
        return month_file
    f = open(pkl_file, 'rb')
    pkl_raw_data = pickle.load(f)
    if len(pkl_raw_data) == 0:
        raise ValueError('pkl file is null!')
    else:
        pkl_data = pkl_raw_data[0]
    f.close()

    
    conv_filename = "../data/pre_training/reddit/data/" + month_file + ".convos.txt"
    # conv_filename = "../data/pre_training/reddit/data/" + month_file + ".convos.txt"
    if not os.path.isfile(conv_filename):
        print(month_file + " conv file not exist, exit!")
        return month_file
    print("processing: " + month_file)
    try:
        conv_df = pd.read_csv(conv_filename, sep="\t", header=None, on_bad_lines='skip')
    except:
        print(month_file + "conv file error, exit!")
        return month_file
    conv_df.columns = ["hash_value", "subreddit_name", "conv_id", "response_score", "dialog_turn_num", "conv_context", "response"]
    conv_df = conv_df.sort_values(by=["conv_id", "dialog_turn_num", "response_score"], ascending=[True, True, False])
    conv_df = conv_df.reset_index(drop=True)
    print(month_file + " conv_df_len: " + str(len(conv_df)))
    if len(conv_df) == start_index:
        print("this conv file has been fully processed, exit!")
        return month_file
    list_context = []
    list_entities = []
    list_response = []
    i_dp_count = 0
    for index, row in tqdm(conv_df[start_index:].iterrows()):
        i_dp_count += 1
        entities_in_context = []
        if  row["dialog_turn_num"] % 2 == 0:
            entities_in_context = []
            dialog_context = _get_context(row["conv_context"])
            response = {"message": _clean_text(row["response"]), "role": "agent"}
            if not _profanity_check(response["message"]) and not _url_check(response["message"]):
                wiki_URI = _get_wiki_uri_name(pkl_data, row["conv_id"])
                if wiki_URI:
                    entities_in_context = _get_entities_of_uri(dialog_context, response, wiki_URI)
                    if not entities_in_context:
                        continue
                    list_context.append(copy.deepcopy(dialog_context))
                    list_response.append(response)
                    list_entities.append(copy.deepcopy(entities_in_context))
        # saving every 2000 datapoints
        if i_dp_count%2000 == 0:
            print("saving results... finish " + str(i_dp_count) + "data points")
            output = []
            for index, elem in enumerate(list_context):
                output.append({"context": list_context[index], "entities": list_entities[index],"response": list_response[index]})

            with open(pre_train_file, 'a+') as fp:
                for elem in output:
                    json.dump(elem, fp)
                    fp.write('\n')
            with open(config_file, 'w') as config_json:
                json.dump({'start_index': (start_index + i_dp_count)}, config_json)
            list_context = []
            list_entities = []
            list_response = []


    output = []
    for index, elem in enumerate(list_context):
        output.append({"context": list_context[index], "entities": list_entities[index],"response": list_response[index]})

    with open(pre_train_file, 'a+') as fp:
        for elem in output:
            json.dump(elem, fp)
            fp.write('\n')
    with open(config_file, 'w') as config_json:
        json.dump({'start_index': (start_index + i_dp_count)}, config_json)
    return month_file

def cb_finish(month_file):
    print(month_file + " finished!")

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--year', type=int, required=True,
                    help='An optional integer argument')
    parser.add_argument('--month', type=int, required=True,
                    help='An optional integer argument')
    args = parser.parse_args()
    n_year = args.year
    n_month = args.month
    month = list(range(n_month,n_month+1))
    year = list(range(n_year,n_year+1))
    list_file = []
    for y in year:
        for m in month:
            if m < 10:
                list_file.append(str(y) + "-0" + str(m))
            else:
                list_file.append(str(y) + "-" + str(m))
    for file in list_file:
        process_dataset(file)
    
    # data = []
    # with open('/mnt/output/pre_training/reddit/data/pretrain_data/pretrain_reddit_dbpedia_2011-10.json', 'r',encoding='utf-8') as f:
    #     for line in f:
    #         data.append(json.loads(line))
    # print(type(data))
    # print(len(data))
    # print(data[:10])
    # pool = Pool(os.cpu_count() - 1)
    # # pool.map(process_dataset, list_file)
    # for month_file in list_file:
    #     pool.apply_async(process_dataset, args = (month_file, ), callback = cb_finish)
    # pool.close()
    # pool.join()
    print("all finished!")