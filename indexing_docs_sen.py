import pandas as pd
import numpy as np

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


from rank_bm25 import BM25Okapi,BM25Plus,BM25L

docs_dfs=pd.read_csv("/home/ricky/Documents/PhDproject/result/trec/docs_top_100.csv",sep='\t')

qids = np.unique(docs_dfs.qid.values)
sens_all=[]
for qid in qids:
    docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
    for ii, doc_rows in docs_tops.iterrows():
        print(qid,ii)
        query=doc_rows['query']
        texts=doc_rows['text']
        corpus=[]
        for i,sen in enumerate(texts.split(".")):
            sen_clean=clean_en_text(sen)
            if len(sen_clean.split())>2 and  sen_clean not in corpus:
                corpus.append(sen_clean)

        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        topical_sens=bm25.get_top_n(tokenized_query, corpus, n=3)
        sens=''
        for topical_sen in topical_sens:
            sens+=topical_sen+"\t 1,"
        sens_all.append(sens)
docs_dfs['top_sens']=sens_all

docs_dfs.to_csv('./docs/docs_all_sens_bm25.csv', index=None, sep=';')
