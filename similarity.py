import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
#Encoding: bert-base-nli-mean-tokens

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


def get_vectors(dfs):
    vecs = []
    for ii, rows in dfs.iterrows():
        chuck_vecs = []
        texts = clean_en_text(rows['text'])
        simis=[]
        texts=texts.split('.')
        for i in range(0, len(texts)):
            sen_embeddings = model.encode(texts[i])
            query_embeddings = model.encode(rows['query'])
            simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
            simis.append([texts[i],simi])
            chuck_vecs.append(sen_embeddings*simi)
        vecs.append(np.mean(chuck_vecs, axis=0))
    dfs['vectors'] = vecs
    return dfs



def get_score_n(journal_dfs_vec,docs_dfs_vec,top_n=10,d_top=100):
    qids = np.unique(journal_dfs_vec.qid.values)
    similarity = []
    for qid in qids:
        journal_tops = journal_dfs_vec.loc[journal_dfs_vec['qid'] == qid].sort_values(by=['rank']).head(top_n)
        docs_tops = docs_dfs_vec.loc[docs_dfs_vec['qid'] == qid].sort_values(by=['rank'])
        for ii, doc_rows in docs_tops.iterrows():
            for jj, journal_rows in journal_tops.iterrows():
                similarity.append([qid, doc_rows['docno'], journal_rows['docno'],
                                   cosine_similarity([doc_rows['vectors']], [journal_rows['vectors']])[0][0]])

    similarity_df = pd.DataFrame(similarity, columns=['qid', 'docno', 'j_docno', 'scores'])
    similarity_path=f'''./experiments/dtop{d_top}_jtop{top_n}'''
    mkdir_p(similarity_path)
    similarity_df.to_csv('%s/w_similarity_score_sw_biobert.csv'%similarity_path, index=None, sep='\t')
    return similarity_df

#load dfs
journal_dfs=pd.read_csv("/home/ricky/Documents/PhDproject/result/trec/journal_wnum_top_30.csv",sep='\t')
docs_dfs=pd.read_csv("/home/ricky/Documents/PhDproject/result/trec/docs_top_100.csv",sep='\t')

journal_dfs_vec=get_vectors(journal_dfs)
docs_dfs_vec=get_vectors(docs_dfs)

simi_score=get_score_n(journal_dfs_vec,docs_dfs_vec)



