import pyterrier as pt
import os
import sys
import collections
import pandas as pd
import numpy as np
import pickle
import pytrec_eval
if not pt.started():
  pt.init()

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

wic_data = './all_journal_content.csv'
index_path='/home/ricky/Documents/PhDproject/indexs/Goodit/journal_index_wnumbers'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()
df_docs['contents'] = df_docs.apply(lambda x: clean_en_text(x['contents']), axis=1)
df_docs.columns = ['title', 'text', 'citation', 'views', 'j_type', 'docno']


user_study_df=pd.read_csv('docs_user_study_sen_j_id.csv', sep='\t')
import ast
j_id=[]
for ii,j in user_study_df.iterrows():
    sentence_dicts=ast.literal_eval(j['sentence_dicts'])
    if sentence_dicts:

        sentence_dicts=list(sentence_dicts.items())[0][1]
        for sentence_dict in sentence_dicts:
            if sentence_dict[-1] not in j_id:
                j_id.append(sentence_dict[-1])

from scholarly import ProxyGenerator, scholarly
def get_info(query):
    pg = ProxyGenerator()
    pg.FreeProxies()
    scholarly.use_proxy(pg)

    # Now search Google Scholar from behind a proxy
    search_query = scholarly.search_pubs(query)
    search_result=next(search_query)
    return ",".join(search_result['bib']['author']),search_result['bib']['pub_year'],search_result['bib']['venue'],search_result['pub_url']


import time
journal={}
for jj,j_ids in enumerate(j_id):
    print(jj)
    jounral_title = df_docs.loc[df_docs['docno'] == j_ids]['title'].values[0]
    journal[j_ids]={}
    journal[j_ids]['title']=jounral_title
    time.sleep(2)
    author,pub_year,venue,url=get_info(jounral_title)
    journal[j_ids]['author']=author
    journal[j_ids]['pub_year'] = pub_year
    journal[j_ids]['venue'] = venue
    journal[j_ids]['url'] = url
    journal[j_ids]['citation']=author+" "+ pub_year+"."+jounral_title+"."+venue

journal[j_id[10]]['citation']='Janiaud, P., Axfors, C., Schmitt, A. M., Gloy, V., Ebrahimi, F., Hepprich, M., ... & Hemkens, L. G. (2021). Association of convalescent plasma treatment with clinical outcomes in patients with COVID-19: a systematic review and meta-analysis. Jama, 325(12), 1185-1195.'


journal_df=[]
journal_list=list(journal.items())
for i in journal_list:
    j_docno=i[0]
    j_title=list(i[1].items())[0][1]
    j_citation=list(i[1].items())[-1][1]
    j_author=list(i[1].items())[1][1]
    j_year = list(i[1].items())[2][1]
    j_url = list(i[1].items())[-2][1]

    journal_df.append([j_docno,j_title,j_citation,j_author,j_year,j_url])

journal_df=pd.DataFrame(journal_df,columns=['docno','title','cite','author','year','url'])
journal_df.to_csv('journal_citation.csv',sep='\t',index=False)


import pandas as pd

user_study_df=pd.read_csv('docs_user_study_sen_j_id.csv', sep='\t')
journal_df=pd.read_csv('journal_citation.csv',sep='\t')

j_id=[]
import ast
for ii,j in user_study_df.iterrows():
    sentence_dicts=ast.literal_eval(j['sentence_dicts'])
    sentence_dicts=list(sentence_dicts.items())[0][1]
