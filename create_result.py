import pandas as pd
import numpy as np


# simi_score=pd.read_csv('./experiments/dtop100_jtop10/clef2020_similarity_biobert.csv', sep='\t')
simi_score=pd.read_csv("./experiments/dtop100_jtop10/trec_1M_similarity_biobert.csv",sep='\t')
#topicality_dfs = pd.read_csv("/home/ricky/Documents/PhDproject/result/trec/docs_top_100.csv", sep='\t')
topicality_dfs=pd.read_csv("/home/ubuntu/rupadhyay/CREDPASS/docs/TREC2020_BM25_clean_100.csv",sep='\t')
#CLEF
# topicality_dfs = pd.read_csv("/tmp/pycharm_project_631/docs/clef2020_docs.csv", sep=';')

def convert_scoreto_column(simi_score):
    qids = np.unique(simi_score.qid.values)
    fixed_dfs = []
    for qid in qids:
        qid_dfs = simi_score.loc[simi_score['qid'] == qid]
        qid_docs = np.unique(simi_score.loc[simi_score['qid'] == qid].docno.values)
        for qid_doc in qid_docs:
            docs_list=[]
            docs_list.append(qid)
            docs_list.append(qid_doc)
            docs_list+=list(qid_dfs.loc[qid_dfs['docno'] == qid_doc].scores.values)
            fixed_dfs.append(docs_list)
    s=['s%s'%i for i in range(10)]
    fixed_dfs=pd.DataFrame(fixed_dfs,columns=['qid','docno']+s)
    return fixed_dfs


# def get_final_cred_score_weights(simi_score):
#     fixed_dfs = convert_scoreto_column(simi_score)
#
#     fixed_dfs = fixed_dfs.drop(['docno', 'qid'], axis=1)
#     fixed_dfs = fixed_dfs.to_numpy(dtype=np.float64)
#     # Load Criterion Type: 'max' or 'min'
#     criterion_type = ['max', 'max', 'max', 'max', 'min', 'min', 'min', 'min', 'min', 'min']
#     #criterion_type=['max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max', 'max','max', 'max', 'min', 'min', 'min', 'min', 'min', 'min', 'min', 'min']
#     weights = critic_method(fixed_dfs, criterion_type)
#     return weights


def get_averaged_score(simi_score):
    qids = np.unique(simi_score.qid.values)
    #weights_10 = get_final_cred_score_weights(simi_score)
    weights_10 = [0.25, 0.15, 0.125, 0.125, 0.1, 0.05, 0.05, 0.05,0.05,0.05]
    average_dfs = []
    for qid in qids:
        qid_dfs = simi_score.loc[simi_score['qid'] == qid]
        qid_docs = np.unique(simi_score.loc[simi_score['qid'] == qid].docno.values)
        for qid_doc in qid_docs:
            #weights_20 = [0.12,0.1,0.06,0.06,0.05,0.05,0.05,0.05,0.05,0.05,0.04,0.04,0.04,0.04,0.02,0.02,0.02,0.02,0.01,0.01]

            score = qid_dfs.loc[qid_dfs['docno'] == qid_doc].scores*weights_10
            score_av = np.sum(score)
            average_dfs.append([qid, "Q0", qid_doc, score_av])
    ave_cred = pd.DataFrame(average_dfs, columns=['qid', 'Q0', 'docno', 'c_score'])
    return ave_cred

def get_weights():
    weights=[0.4,0.6]
    return weights

def get_zscore(final_df):
    qid_dfs = []
    qids = np.unique(final_df.qid.values)
    for qid in qids:
        qid_df = final_df.loc[final_df['qid'] == qid]
        qid_df['score_zscore'] = (qid_df['score'] - qid_df['score'].min()) / (
                    qid_df['score'].max() - qid_df['score'].min())
        qid_dfs.append(qid_df)
    return pd.concat(qid_dfs)


def get_result_df(simi_score):
    ave_cred=get_averaged_score(simi_score)

    final_df=ave_cred.merge(topicality_dfs,on=['qid','docno'])

    final_df.to_csv('//tmp/pycharm_project_631/experiments/dtop100_jtop10/combined_df_biobert.csv',sep='\t',index=None)

    normalized_df=get_zscore(final_df)


    weights=get_weights()
    final_df['combined_score']=final_df['score']*weights[0]+final_df['c_score']*weights[1]

    qids = np.unique(simi_score.qid.values)
    sorted_dfs=[]
    for qid in qids:
        if qid!=28:
            qid_df=final_df.loc[final_df['qid']==qid]
            sorted_qid_df=qid_df.sort_values('combined_score',ascending=False).reset_index()
            sorted_qid_df['n_rank']=1
            for i in sorted_qid_df.index:
                sorted_qid_df.at[i,'n_rank']=i
            sorted_dfs.append(sorted_qid_df)
    sorted_qid_df_concat=pd.concat(sorted_dfs)


    result_df=sorted_qid_df_concat[['qid','docno','n_rank','combined_score']]
    result_df['Q0']=0
    result_df=result_df[['qid','Q0','docno','n_rank','combined_score']]
    result_df.columns=['qid','Q0','docno','rank','score']
    result_df['experiment']='trec_1M_wa_d100_j10'
    result_df.to_csv('./result/40_60_biobert_simi_wa_d100_j10_trec_20.csv',sep=' ',index=None,header=None)
    return result_df


dd=get_result_df(simi_score)
