import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('pritamdeka/S-Biomed-Roberta-snli-multinli-stsb')
import numpy as np
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


# def get_vectors(dfs):
#     vecs = []
#     for ii, rows in dfs.iterrows():
#         chuck_vecs = []
#         texts = clean_en_text(rows['text'])
#         simis=[]
#         texts=texts.split('.')
#         for i in range(0, len(texts)):
#             sen_embeddings = model.encode(texts[i])
#             query_embeddings = model.encode(rows['query'])
#             simi=cosine_similarity([query_embeddings,sen_embeddings])[0][1]
#             simis.append([texts[i],simi])
#             chuck_vecs.append(sen_embeddings*simi)
#         vecs.append(np.mean(chuck_vecs, axis=0))
#     dfs['vectors'] = vecs
#     return dfs



def get_score_n(journal_dfs,docs_dfs,root_path,top_n=10,d_top=100):
    qids = np.unique(docs_dfs.qid.values)
    similarity = []
    for qid in qids:
        print(qid)
        journal_tops = journal_dfs.loc[journal_dfs['qid'] == qid].sort_values(by=['rank']).head(top_n)
        docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
        for ii, doc_rows in docs_tops.iterrows():
            doc_sens = [(i.split('\t')[0],float(i.split('\t')[1])) for i in doc_rows['top_sens'].split(",") if len(i)>0]
            doc_sens_sorted=sorted(doc_sens, key=lambda t: t[1], reverse=True)
            for doc_sen in doc_sens_sorted:
                sens_evi=''
                if doc_sen[-1]>0.4:
                    doc_sen_vec = model.encode(doc_sen[0])
                    for jj, journal_rows in journal_tops.iterrows():
                        texts = journal_rows['text'].split('.')
                        for sen in texts:
                            if len(sen.split(" "))>5:
                                jou_sen_vec=model.encode(sen)
                                simi = cosine_similarity([doc_sen_vec, jou_sen_vec])[0][1]
                                if simi>0.0:
                                    sens_evi+='%s\t %s\t %s,'%(doc_sen[0],sen,simi)
                        journal_evi_sorted = sorted([('%s\t %s' % (i.split('\t')[0], i.split('\t')[1]), float(i.split('\t')[-1])) for i
                                        in sens_evi.split(",") if len(i) > 0], key=lambda t:t[1],reverse=True)[:10]

                        similarity.append([qid, doc_rows['docno'], journal_rows['docno'],journal_evi_sorted,doc_rows['rank']])

    similarity_df = pd.DataFrame(similarity, columns=['qid', 'docno', 'j_docno', 'scores','rank'])
    similarity_path=f'''{root_path}/experiments/dtop{d_top}_jtop{top_n}'''
    mkdir_p(similarity_path)
    similarity_df.to_csv('%s/ner_manual_sens_similarity_score_sw_biobert.csv'%similarity_path, index=None, sep='\t')
    return similarity_df

#load dfs
root_path='/tmp/pycharm_project_631/'
journal_dfs=pd.read_csv("%s/docs/journal_wnum_top_30.csv"%root_path,sep='\t')
#docs_dfs=pd.read_csv("%s/docs/docs_all_top_sen_ner.csv"%root_path,sep=';')
docs_dfs=pd.read_csv("%s/docs/docs_all_top_sen_ner_manual.csv"%root_path,sep=';')

#
#docs_dfs_1=docs_dfs[docs_dfs['qid']==1]
#journal_dfs_vec=get_vectors(journal_dfs)
#docs_dfs_vec=get_vectors(docs_dfs_1)
docs_dfs.dropna(inplace=True)
simi_score=get_score_n(journal_dfs,docs_dfs,root_path)
# #
#
import pandas as pd
import numpy as np
import ast
# simi_score=pd.read_csv('./experiments/dtop100_jtop10/Bm25_sens_similarity_score_sw_biobert.csv',sep='\t')
simi_score=pd.read_csv('./experiments/dtop100_jtop10/ner_manual_sens_similarity_score_sw_biobert.csv',sep='\t')
#
#


qids = np.unique(simi_score.qid.values)
cred_score = []

simi_score['rank']=simi_score['rank'].astype(int)
for qid in qids:
    qid_simi_score = simi_score.loc[(simi_score['qid'] == qid) & (simi_score['rank']<100)].sort_values(by=['rank'])
    docnos_index=np.unique(qid_simi_score.docno.values,return_index=True)[1]
    docnos=[qid_simi_score.docno.values[index] for index in sorted(docnos_index)]
    docnos=list(set(docnos))
    for docno in docnos:
        docs_specific=qid_simi_score.loc[qid_simi_score['docno']==docno]
        docs_score = []
        journs_index=np.unique(docs_specific.j_docno.values,return_index=True)[1]
        journs=[docs_specific.j_docno.values[j_index] for j_index in sorted(journs_index)]
        for jour in journs[:1]:
            jours_specifics=docs_specific.loc[docs_specific['j_docno']==jour].head(1)
            for ii,rows in jours_specifics.iterrows():
                sentences=ast.literal_eval(rows['scores'])
                #sentences=rows['scores']
                if len(sentences)>0:
                    for sentence in sentences[:1]:
                        docs_score.append(sentence[1])
                else:
                    docs_score.append(0)
        cred_score.append([qid,docno,np.mean(docs_score, axis=0)])


qrels="./qrels/misinfo-qrels.2aspects.useful-credible"
qrels_df=pd.read_csv(qrels,sep=' ',header=None,names=['qid','Q0','docno','top','cred'])
#qrels_df=qrels_df.loc[qrels_df['qid']==1]
qrels_df=qrels_df[['qid','docno','cred']]
cred_score_df=pd.DataFrame(cred_score,columns=['qid','docno','cred'])
inner=qrels_df.merge(cred_score_df,on=['qid','docno'])

from sklearn.metrics import roc_curve, auc

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


# Add prediction probability to dataframe

# Find optimal probability threshold
threshold = Find_Optimal_Cutoff(inner['cred_x'].values, inner['cred_y'].values)
print(threshold)

# Find prediction to the dataframe applying threshold
inner['cred_y_pred'] = inner['cred_y'].map(lambda x: 1 if x > threshold[0] else 0)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(inner['cred_x'],inner['cred_y_pred']))

from sklearn.metrics import f1_score
print(f1_score(inner['cred_x'], inner['cred_y_pred']))

from sklearn.metrics import fowlkes_mallows_score
print(fowlkes_mallows_score(inner['cred_x'],inner['cred_y_pred']))

from sklearn.metrics import roc_auc_score
print(roc_auc_score(inner['cred_x'],inner['cred_y']))

##