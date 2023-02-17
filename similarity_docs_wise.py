import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb').to('cuda')
#Encoding: bert-base-nli-mean-tokens
# pritamdeka/S-PubMedBert-MS-MARCO-SCIFACT

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


def get_vectors(dfs,window=20):
    vecs = []
    for ii, rows in dfs.iterrows():
        chuck_vecs = []
        texts = clean_en_text(rows['text'])
        simis=[]
        for i in range(0, len(texts.split()), 512-window):
            texts_512 = " ".join(texts.split()[i:i + 512-window])
            sen_embeddings = model.encode(texts_512)
            #query_embeddings = model.encode(rows['query'])
            #simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
            #simis.append(simi)
            chuck_vecs.append(sen_embeddings)
        vecs.append(np.mean(chuck_vecs, axis=0))
    dfs['vectors'] = vecs
    return dfs

#
# def get_vectors(dfs):
#     vecs = []
#     sens_all=[]
#     for ii, rows in dfs.iterrows():
#         print(ii)
#         chuck_vecs = []
#         texts = clean_en_text(rows['text'])
#         simis=[]
#         sens=''
#         texts=texts.split('.')
#         for i in range(0, len(texts)-1):
#             sen_embeddings = model.encode([texts[i]+" "+texts[i+1]])
#             query_embeddings = model.encode(rows['query'])
#             simi=cosine_similarity([query_embeddings,sen_embeddings[0]])[0][1]
#             if simi>0.41:
#               sens+=texts[i]+" "+texts[i+1]+"\t %s,"%simi
#             simis.append([texts[i],simi])
#             chuck_vecs.append(sen_embeddings*simi)
#         vecs.append(np.mean(chuck_vecs, axis=0))
#         sens_all.append(sens)
#     dfs['vectors'] = vecs
#     dfs['top_sens'] = sens_all
#     return dfs

#load dfs
#docs_dfs=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/docs/Clef2020_BM25_100.csv",sep='\t')
docs_dfs=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv",sep='\t')
qids = np.unique(docs_dfs.qid.values)
sens_all = []
dfs_tops=[]
for qid in qids:
    dfs_tops.append(docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank']).head(100))
dfs_tops=pd.concat(dfs_tops).reset_index()
dfs_tops.drop(['index'],inplace=True,axis=1)

#docs_dfs_1=docs_dfs[docs_dfs.qid==1]
docs_dfs_vec=get_vectors(dfs_tops)
docs_dfs_vec.to_csv('./docs/trec_1M_docs.csv', index=None, sep=';')



