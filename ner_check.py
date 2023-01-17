from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import numpy as np
import ast
from difflib import SequenceMatcher
simi_score=pd.read_csv('./experiments/dtop100_jtop10/Bm25_sens_similarity_score_sw_biobert.csv',sep='\t')

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu

def get_entity_name(entities):
    if len(entities)>0:
        return_word=[]
        for entity in entities:
            if entity['entity_group']=='Medication':
                if entity['word'] not in return_word:
                    return_word.append(entity['word'])
        if len(return_word)>0:

            return return_word[0]
        else:
            return None
    else:
        return None

qids = np.unique(simi_score.qid.values)
cred_score = []
for qid in qids:
    print(qid)
    qid_simi_score = simi_score.loc[simi_score['qid'] == qid].sort_values(by=['rank'])
    docnos_index=np.unique(qid_simi_score.docno.values,return_index=True)[1]
    docnos=[qid_simi_score.docno.values[index] for index in sorted(docnos_index)]
    for docno in docnos:
        docs_specific=qid_simi_score.loc[qid_simi_score['docno']==docno]
        docs_score = []
        for ii,rows in docs_specific.iterrows():
            sentences=ast.literal_eval(rows['scores'])
            if len(sentences) > 0:
                for sentence in sentences[:5]:
                    texts=sentence[0].split("\t")
                    doc_sen_entity_name=get_entity_name(pipe(texts[0]))
                    arti_sen_entity_name=get_entity_name(pipe(texts[1]))
                    if doc_sen_entity_name and arti_sen_entity_name:
                        if doc_sen_entity_name==arti_sen_entity_name:
                            docs_score.append(sentence[1])
                        elif SequenceMatcher(None, doc_sen_entity_name,arti_sen_entity_name).ratio()>0.95:
                            docs_score.append(sentence[1]*0.25)
                        else:
                            docs_score.append(sentence[1]*0.1)
                    else:
                        docs_score.append(0)
            else:
                docs_score.append(0)
        cred_score.append([qid, docno, np.mean(docs_score, axis=0)])


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
print(f1_score(inner['cred_x'], inner['cred_y_pred'], average='macro'))