import pandas as pd
import time
from semanticscholar import SemanticScholar
sch = SemanticScholar(api_key='chcsWY6XL5a3fp4Vgolsu82vbc629SSP7DIs9MFA')
wic_data = '/tmp/pycharm_project_631/all_journal_content.csv'
df_docs = pd.read_csv(wic_data, sep='\t', index_col=0, lineterminator='\n').dropna()

apa_citations=[]
for ii,df_rows in df_docs.iterrows():
    print(ii)
    try:
        title=df_rows['title']
        results = sch.search_paper(title)
        if results:
            abstracts=results[0].abstract
            # apa_citations.append([df_rows['id'],df_rows['title'],abstracts,df_rows['contents'],df_rows['citation'],df_rows['view'],df_rows['j_type']])
            apa_citations.append(abstracts)
        else:
            apa_citations.append(None)
    except:
        print("Error")
        apa_citations.append(None)

df_docs['abstract']=apa_citations
df_docs.to_csv('/tmp/pycharm_project_631/all_journal_content_abstract.csv',sep='\t')