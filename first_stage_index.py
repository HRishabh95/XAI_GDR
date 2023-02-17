import os
import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init()
import sys
os.environ['JAVA_HOME'] = "/usr/lib/jvm/java-1.11.0-openjdk-amd64/"
import re
import string
PUNCTUATIONS = string.punctuation
PUNCTUATIONS = PUNCTUATIONS.replace(".",'')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?%\'`]", " ", text)
  text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def make_docs_cerche(dfs):
  docs=[]
  for ii,df in dfs.iterrows():
    if df['text']:
      docs.append({'id':df['docno'],'title':'','article':clean_en_text(df['text'])})
  return docs

config={'TREC':{'file_path':'/home/ubuntu/rupadhyay/CREDPASS/TREC2020_1M_labeled_clean.csv',
                'index_path':'/home/ubuntu/rupadhyay/CREDPASS/trec2020_bm25',
                'topics':'/home/ubuntu/rupadhyay/dataset/TREC/topics.csv'},
        'CLEF':{'file_path':'/home/ubuntu/rupadhyay/CREDPASS/Clef2020_1M_labeled_clean.csv',
                'index_path':'/home/ubuntu/rupadhyay/CREDPASS/clef2020_bm25',
                'topics':'/home/ubuntu/rupadhyay/CREDPASS/clef_topics.csv'}}

data_set='CLEF'
indexing=False

indexref3 = pt.IndexRef.of(f'''{config[data_set]['index_path']}/data.properties''')
BM251 = pt.BatchRetrieve(indexref3, num_results=100, controls = {"wmodel": "BM25"})


topics=pt.io.read_topics(config[data_set]['topics'],format='singleline')
results=BM251.transform(topics)

results['Q0']=0
result=results[['qid','Q0','docno','rank','score']]
# qrels_path='/home/ubuntu/rupadhyay/CREDPASS/clef_qrels_top.csv'
# qrels = pt.io.read_qrels(qrels_path)
# eval = pt.Utils.evaluate(results,qrels,metrics=["ndcg"], perquery=True)

docs=pd.read_csv(config[data_set]['file_path'],sep='\t')
docs=docs[['docno','text']]
merged_results=pd.merge(results,docs,on=['docno'])

merged_results.to_csv('/home/ubuntu/rupadhyay/CREDPASS/docs/Clef2020_BM25_100.csv',sep='\t',index=None)
