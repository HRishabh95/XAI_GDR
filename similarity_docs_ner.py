import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
#Encoding: bert-base-nli-mean-tokens
# pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT
from transformers import pipeline
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForTokenClassification
ner_model='d4data/biomedical-ner-all'
tokenizer = AutoTokenizer.from_pretrained(ner_model)
model_ner = AutoModelForTokenClassification.from_pretrained(ner_model)
pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu

import errno
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

import re
import string
PUNCTUATIONS = string.punctuation.replace('.','')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

#
# def get_vectors(dfs,window=20):
#     vecs = []
#     for ii, rows in dfs.iterrows():
#         chuck_vecs = []
#         texts = clean_en_text(rows['text'])
#         simis=[]
#         for i in range(0, len(texts.split()), 512-window):
#             texts_512 = " ".join(texts.split()[i:i + 512-window])
#             sen_embeddings = model.encode(texts_512)
#             query_embeddings = model.encode(rows['query'])
#             simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
#             simis.append(simi)
#             chuck_vecs.append(sen_embeddings*simi)
#         vecs.append(np.mean(chuck_vecs, axis=0))
#     dfs['vectors'] = vecs
#     return dfs

def get_entity_name(entities):
    if len(entities)>0:
        return_word=[]
        for entity in entities:
            if entity['entity_group']=='Medication':
                if entity['word'] not in return_word:
                    return_word.append(entity['word'])
        if len(return_word)>0:

            return return_word[0]
        else:
            return None
    else:
        return None

def get_medication_query(texts):
    texts=texts.replace('covid 19','')
    texts=" ".join(texts.split(" ")[:-2])
    return texts

def get_vectors(dfs):
    vecs = []
    sens_all=[]
    for ii, rows in dfs.iterrows():
        print(ii)
        chuck_vecs = []
        texts = clean_en_text(rows['text'])
        simis=[]
        sens=''
        texts=texts.split('.')
        for i in range(0, len(texts)):
            sen_embeddings = model.encode(texts[i])
            query_embeddings = model.encode(rows['query'])
            query_sen_entity_name = get_entity_name(pipe(rows['query']))
            arti_sen_entity_name = get_entity_name(pipe(texts[i]))
            simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
            if query_sen_entity_name and arti_sen_entity_name:
                if query_sen_entity_name==arti_sen_entity_name:
                    sens+=texts[i]+"\t %s,"%(float(simi))
                elif SequenceMatcher(None, query_sen_entity_name, arti_sen_entity_name).ratio()>0.9:
                    sens += texts[i] + "\t %s," % (float(simi)*0.65)
                else:
                    sens += texts[i] + "\t %s," % (float(simi) * 0.2)
            else:
                query_sen_entity_name=get_medication_query(rows['query'])
                if query_sen_entity_name in texts[i]:
                    sens+=texts[i]+"\t %s,"%(float(simi))
                else:
                    sens += texts[i] + "\t %s," % (float(simi) * 0.4)
            simis.append([texts[i],simi])
            chuck_vecs.append(sen_embeddings*simi)
        vecs.append(np.mean(chuck_vecs, axis=0))
        sens_all.append(sens)
    dfs['vectors'] = vecs
    dfs['top_sens'] = sens_all
    return dfs

#load dfs
docs_dfs=pd.read_csv("./docs/docs_top_100.csv",sep='\t')
# qids = np.unique(docs_dfs.qid.values)
# sens_all = []
# dfs_tops=[]
# for qid in qids:
#     dfs_tops.append(docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank']))
# dfs_tops=pd.concat(dfs_tops).reset_index()
# dfs_tops.drop(['index'],inplace=True,axis=1)

#docs_dfs_1=docs_dfs[docs_dfs.qid==1]
docs_dfs_vec=get_vectors(docs_dfs)
docs_dfs_vec.to_csv('./docs/docs_all_top_sen_ner_manual.csv', index=None, sep=';')



