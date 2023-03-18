from trectools import TrecQrel, procedures
from scipy.stats import hmean



def get_ndcg_p(qrels_file):
    qrels = TrecQrel(qrels_file)
    runs=procedures.list_of_runs_from_path('/home/ricky/Documents/PhDproject/Project_folder/pseudo_journals_Goodit/wISE_result','*.csv')
    results = procedures.evaluate_runs(runs, qrels, per_query=True)
    top_NDCG_10,top_P_10=[],[]
    for result in results:
        result=result.data
        top_NDCG_10.append(result.loc[result['metric']=='NDCG_10']['value'].dropna().values.mean())
        top_P_10.append(result.loc[result['metric']=='map']['value'].dropna().values.mean())
    return top_P_10,top_NDCG_10



qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels.useful.wise"
top_p_10,top_ndcg_10=get_ndcg_p(qrels_file)
qrels_file = "/home/ricky/Documents/PhDproject/Project_folder/2020-derived-qrels/misinfo-qrels.cred.wise"
cred_p_10,cred_ndcg_10=get_ndcg_p(qrels_file)
for i in range(len(top_p_10)):
    MM_NDCG=hmean([top_ndcg_10[i],cred_ndcg_10[i]])
    MM_P_10=hmean([top_p_10[i],cred_p_10[i]])
    print(MM_NDCG,MM_P_10)



## Top5
## Baseline 0.83 0.87
## Model 1 0.8315 0.8755
## Model 2 0.8326 0.8778
## Model 3 0.8401 0.8897
## Model 4 0.849 0.8977


## Top15
## Baseline 0.78 0.84
## Model 1 0.784 0.8454
## Model 2 0.786 0.8478
## Model 3 0.7912 0.8579
## Model 4 0.7956 0.8632


## Top20
## Baseline 0.7705 0.836
## Model 1 0.7712 0.8389
## Model 2 0.7723 0.8403
## Model 3 0.7754 0.8487
## Model 4 0.7765 0.8545