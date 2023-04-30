import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb').to('cuda')
import numpy as np
from nltk.tokenize import sent_tokenize
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




def get_score_n(journal_dfs,docs_dfs,root_path,top_n=10,d_top=100):
    qids = np.unique(docs_dfs.qid.values)
    similarity = []
    for qid in qids:
        print(qid)
        journal_tops = journal_dfs.loc[journal_dfs['qid'] == qid].sort_values(by=['rank']).head(top_n)
        docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
        for ii, doc_rows in docs_tops.iterrows():
            for jj, journal_rows in journal_tops.iterrows():
                simi = cosine_similarity([doc_rows['vectors'], journal_rows['vectors']])[0][1]
                similarity.append([qid, doc_rows['docno'], journal_rows['docno'],simi])

    similarity_df = pd.DataFrame(similarity, columns=['qid', 'docno', 'j_docno', 'scores'])
    similarity_path=f'''{root_path}/experiments/dtop{d_top}_jtop{top_n}_abstract'''
    mkdir_p(similarity_path)
    similarity_df.to_csv('%s/trec_1M_similarity_biobert_fields.csv'%similarity_path, index=None, sep='\t')
    return similarity_df

#load dfs
root_path='/tmp/pycharm_project_631/'
journal_dfs=pd.read_csv("%s/docs/journal_wnum_top_30_trec_fields.csv"%root_path,sep='\t')
docs_dfs=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv",sep='\t')
# docs_dfs=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/docs/Clef2020_BM25_100.csv",sep='\t')

qids = np.unique(docs_dfs.qid.values)
sens_all = []
dfs_tops=[]
for qid in qids:
    dfs_tops.append(docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank']).head(100))
dfs_tops=pd.concat(dfs_tops).reset_index()
dfs_tops.drop(['index'],inplace=True,axis=1)

#docs_dfs_1=docs_dfs[docs_dfs['qid']==1]
journal_dfs_vec=get_vectors(journal_dfs)
docs_dfs=get_vectors(dfs_tops)
#docs_dfs_vec=get_vectors(docs_dfs_1)
docs_dfs.dropna(inplace=True)
simi_score=get_score_n(journal_dfs,docs_dfs,root_path)