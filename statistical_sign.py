from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_rel,f_oneway
# seed the random number generator
import pandas as pd
import numpy as np

data1 = pd.read_csv("/home/ricky/Documents/PhDproject/result/trec/docs_top_100.csv", sep='\t')
data2 = pd.read_csv('/home/ricky/Documents/PhDproject/Project_folder/pseudo_journals_Goodit/result/40_60_biobert_simi_wa_d100_j10.csv',sep=' ',header=None)
data2.columns=list(data1.columns)

def get_zscore(final_df):
    qid_dfs = []
    qids = np.unique(final_df.qid.values)
    for qid in qids:
        qid_df = final_df.loc[final_df['qid'] == qid]
        qid_df['score_zscore'] = (qid_df['score'] - qid_df['score'].min()) / (
                    qid_df['score'].max() - qid_df['score'].min())
        qid_dfs.append(qid_df)
    return pd.concat(qid_dfs)

topicality_dfs=get_zscore(data1)
combined_dfs=get_zscore(data2)

merged_df=topicality_dfs.merge(combined_dfs,on=['qid','docno'])
score_top=merged_df['score_zscore_x'].values
score_com=merged_df['score_zscore_y'].values


# compare samples
stat, p = ttest_rel(score_com, score_top)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')




# compare samples
stat, p = f_oneway(score_com, score_top)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')


import permutation_test as p

p_value = p.permutationtest(list(score_com[:10]), list(score_top[:10]))
