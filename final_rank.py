
import pandas as pd
import numpy as np
import ast
root_path='/tmp/pycharm_project_631/'
journal_dfs=pd.read_csv("%s/docs/journal_wnum_top_30.csv"%root_path,sep='\t')
#docs_dfs=pd.read_csv("%s/docs/docs_all_top_sen_ner.csv"%root_path,sep=';')
docs_dfs=pd.read_csv("%s/docs/gen_docs_func_all_top_sen_ner_manual.csv"%root_path,sep=';')
# simi_score=pd.read_csv('./experiments/dtop100_jtop10/Bm25_sens_similarity_score_sw_biobert.csv',sep='\t')
simi_score=pd.read_csv('./experiments/dtop100_jtop10/gen_ner_func_manual_sens_similarity_score_sw_biobert.csv',sep='\t')
#
#


qids = np.unique(simi_score.qid.values)
cred_score = []

simi_score['rank']=simi_score['rank'].astype(int)
for qid in qids:
    qid_simi_score = simi_score.loc[(simi_score['qid'] == qid) & (simi_score['rank']<51)].sort_values(by=['rank'])
    docnos_index=np.unique(qid_simi_score.docno.values,return_index=True)[1]
    docnos=[qid_simi_score.docno.values[index] for index in sorted(docnos_index)]
    docnos=list(set(docnos))
    for docno in docnos:
        docs_specific=qid_simi_score.loc[qid_simi_score['docno']==docno]
        docs_score = []
        journs_index=np.unique(docs_specific.j_docno.values,return_index=True)[1]
        journs=[docs_specific.j_docno.values[j_index] for j_index in sorted(journs_index)]
        for jour in journs[:3]:
            jours_specifics=docs_specific.loc[docs_specific['j_docno']==jour]
            for ii,rows in jours_specifics.iterrows():
                sentences=ast.literal_eval(rows['scores'])
                #sentences=rows['scores']
                if len(sentences)>0:
                    for sentence in sentences[:3]:
                        docs_score.append(sentence[1])
                else:
                    docs_score.append(0)
        cred_score.append([qid,docno,np.mean(docs_score, axis=0)])

cred_score_df=pd.DataFrame(cred_score,columns=['qid','docno','cred'])
docs_df_cred=docs_dfs.merge(cred_score_df,on=['qid','docno'])

docs_df_cred['f_score']=0.4*docs_df_cred['score']+0.6*docs_df_cred['cred']
#docs_df_cred_sorted = docs_df_cred.sort_values('f_score', ascending=False).reset_index()

sorted_dfs=[]
for qid in qids:
    if qid!=28 and qid!=191001:
        qid_df=docs_df_cred.loc[docs_df_cred['qid']==qid]
        sorted_qid_df=qid_df.sort_values('f_score',ascending=False).reset_index()
        sorted_qid_df['n_rank']=1
        for i in sorted_qid_df.index:
            sorted_qid_df.at[i,'n_rank']=i
        sorted_dfs.append(sorted_qid_df)

docs_df_cred_sorted=pd.concat(sorted_dfs)


result_df=docs_df_cred_sorted[['qid','docno','n_rank','f_score']]
result_df['Q0']=0
result_df=result_df[['qid','Q0','docno','n_rank','f_score']]
result_df.columns=['qid','Q0','docno','rank','score']

result_df['experiment']='add_pass_ner_d50_j10'

result_df.to_csv('./result/passage/wa_t3j_t3s_ner_biobert_simi_add_d50_j10.csv', sep=' ', index=None, header=None)
