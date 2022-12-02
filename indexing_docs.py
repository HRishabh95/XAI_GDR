import pyterrier as pt
import os
import sys
import collections
import pandas as pd
import numpy as np
import pickle
from cleantext import clean
import pytrec_eval
if not pt.started():
  pt.init()

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"

import re
import string
PUNCTUATIONS = string.punctuation.replace(".",'')


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

def trec_generate(f):
  df = pd.DataFrame(f, columns=['docno', 'text'])
  return df


import os

wic_data = '/home/ricky/Documents/PhDproject/dataset/trec/trec_20_wic_top10_en_nd.csv'
index_path='/home/ricky/Documents/PhDproject/indexs/BM25_baseline_nds'
f = pickle.load(open(wic_data, 'rb'))
df_docs = trec_generate(f)

if not os.path.exists(f'''{index_path}/data.properties'''):

  indexer = pt.DFIndexer(index_path, overwrite=True, verbose=True, Threads=8)
  indexer.setProperty("tokeniser",
                      "UTFTokeniser")  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
  indexer.setProperty("termpipelines", "PorterStemmer")  # Removes the default PorterStemmer (English)
  #indexer.setProperty('metaindex.compressed.crop.long',
  #                    'true')  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
  indexref3 = indexer.index(df_docs["text"], df_docs["docno"])
else:
  indexref3 = pt.IndexRef.of(f'''{index_path}/data.properties''')


topics=pt.io.read_topics("/home/ricky/Documents/PhDproject/dataset/trec/topics.csv", format='singleline',tokenise=True)
topics['query']=topics['query'].replace("can",'',regex=True)
BM25 = pt.BatchRetrieve(indexref3, num_results=100,controls={"wmodel": "BM25"}, properties={
  'tokeniser': 'UTFTokeniser',
  'termpipelines': 'Stopwords,PorterStemmer', })

result=BM25.transform(topics)
combined_result=pd.merge(result,df_docs,on='docno')

results=combined_result[['qid','docid','docno','rank','score','text','query']]

filtered_result=results.loc[results['qid']!='28']
filtered_result.to_csv("/home/ricky/Documents/PhDproject/result/trec/docs_top_100.csv",index=None,sep='\t')


### TREC Eval file from pyterrier file
def create_eval_file(df):
  docs_dfs = df[['qid', 'docno', 'rank', 'score']]
  docs_dfs['Q0'] = 'Q0'
  docs_dfs['experiment'] = 'BM25_baseline'
  docs_dfs = docs_dfs[['qid', 'Q0', 'docno', 'rank', 'score', 'experiment']]
  docs_dfs.to_csv("/home/ricky/Documents/PhDproject/Project_folder/pseudo_journals_Goodit/result/BM25_baseline_top_100.csv",
                  sep=' ', index=None, header=None)



