from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI").to('cuda')

import pandas as pd
import numpy as np
import ast
from collections import Counter

# simi_score=pd.read_csv('./experiments/dtop100_jtop10/Bm25_sens_similarity_score_sw_biobert.csv',sep='\t')
simi_score=pd.read_csv('./experiments/dtop100_jtop10/ner_manual_sens_similarity_score_sw_biobert.csv',sep='\t')
#

def get_nli(sentence):
    sent_1,sent_2=sentence[0].split("\t")[0],sentence[0].split("\t")[1]
    text = f"mednli: sentence1: {sent_1} sentence2: {sent_2}"
    encoding = tokenizer.encode_plus(text, padding='max_length', max_length=256, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to('cuda'), encoding["attention_mask"].to('cuda')
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=8,
        early_stopping=True
    )
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return line

def do_consensus(docs_score):
    counters=Counter(docs_score)
    most_common=counters.most_common()
    if len(most_common)>1:
        if most_common[0][0]=='neutral':
            if most_common[1][0]=='entailment':
                return 1
            else:
                return 0
        else:
            if most_common[0][0] == 'entailment':
                return 1
            else:
                return 0
    else:
        if most_common[0][0]=='entailment':
            return 1
        else:
            return 0

qids = np.unique(simi_score.qid.values)
cred_score = []

simi_score['rank']=simi_score['rank'].astype(int)
for qid in qids:
    print(qid)
    qid_simi_score = simi_score.loc[(simi_score['qid'] == qid) & (simi_score['rank']<10)].sort_values(by=['rank'])
    docnos_index=np.unique(qid_simi_score.docno.values,return_index=True)[1]
    docnos=[qid_simi_score.docno.values[index] for index in sorted(docnos_index)]
    docnos=list(set(docnos))
    for docno in docnos:
        docs_specific=qid_simi_score.loc[qid_simi_score['docno']==docno]
        journs_index=np.unique(docs_specific.j_docno.values,return_index=True)[1]
        journs=[docs_specific.j_docno.values[j_index] for j_index in sorted(journs_index)]
        journal_consensus=[]
        for jour in journs:
            docs_score = []
            jours_specifics=docs_specific.loc[docs_specific['j_docno']==jour].drop_duplicates().sort_index()
            for ii,rows in jours_specifics.iterrows():
                sentences=ast.literal_eval(rows['scores'])
                #sentences=rows['scores']
                if len(sentences)>0:
                    for sentence in sentences[:5]:
                        nli=get_nli(sentence)
                        if nli=='entailment':
                            docs_score.append(sentence[1])
                        elif nli=='neutral':
                            docs_score.append(0.5*float(sentence[1]))
                        else:
                            docs_score.append(-1*float(sentence[1]))
        cred_score.append([qid,docno,np.mean(docs_score,axis=0)])



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