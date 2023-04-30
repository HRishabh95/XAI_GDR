import pyterrier as pt
import os
import sys
import collections
import pandas as pd
import numpy as np
import pickle
import pytrec_eval
if not pt.started():
  pt.init(helper_version = "0.0.6")

os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"

import re
import string
PUNCTUATIONS = string.punctuation.replace(".","")


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^0-9a-zA-Z().,!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

wic_data = './all_journal_content_abstract.csv'
index_path='./index/journal_index_wnumbers_abstract_fields'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()
df_docs['contents'] = df_docs.apply(lambda x: clean_en_text(x['contents']), axis=1)
df_docs['abstract'] = df_docs.apply(lambda x: clean_en_text(x['abstract']), axis=1)
df_docs.columns = ['title', 'text', 'citation', 'views', 'j_type', 'docno','abstract']


df_docs.drop_duplicates(inplace=True)

df_dict=df_docs.to_dict('records')

if not os.path.exists(f'''{index_path}/data.properties'''):

  # indexer = pt.DFIndexer(index_path, overwrite=True, verbose=True, Threads=8)
  # indexer.setProperty("tokeniser",
  #                     "UTFTokeniser")  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
  # indexer.setProperty("termpipelines", "PorterStemmer")  # Removes the default PorterStemmer (English)
  # #indexer.setProperty('metaindex.compressed.crop.long',
  # #                    'true')  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
  # indexref3 = indexer.index(df_docs["abstract"], df_docs["docno"])

  indexer = pt.IterDictIndexer(index_path,  overwrite=True, verbose=True, Threads=8)
  indexer.setProperty("tokeniser",
                      "UTFTokeniser")  # Replaces the default EnglishTokeniser, which makes assumptions specific to English
  indexer.setProperty("termpipelines", "PorterStemmer")  # Removes the default PorterStemmer (English)
  indexer.setProperty('metaindex.compressed.crop.long',
                     'true')  # Replaces the default EnglishTokeniser, which makes assumptions specific to English

  indexref3 = indexer.index(df_dict,fields=['text','abstract'],meta=['docno'])
else:
  indexref3 = pt.IndexRef.of(f'''{index_path}/data.properties''')


topics=pt.io.read_topics("/home/ubuntu/rupadhyay/CREDPASS/trec_topics.csv", format='singleline',tokenise=True)
# topics=pt.io.read_topics("/home/ubuntu/rupadhyay/CREDPASS/clef_topics.csv", format='singleline',tokenise=True)

#topics=pd.read_csv("/home/ricky/Documents/PhD project/dataset/trec/topics_des.csv",sep=' ',index_col=0)
#topics.to_csv("/home/ricky/Documents/PhD project/dataset/trec/topics.csv",header=None,index=None,sep=' ')
BM25 = pt.BatchRetrieve(indexref3, num_results=30,controls={"wmodel": "BM25F"}, properties={
  'tokeniser': 'UTFTokeniser',
  'termpipelines': 'Stopwords,PorterStemmer', })

result=BM25.transform(topics)
combined_result=pd.merge(result,df_docs,on='docno')

results=combined_result[['qid','query','docid','docno','rank','score','text','abstract']]

results.to_csv("./docs/journal_wnum_top_30_trec_fields.csv",index=None,sep='\t')