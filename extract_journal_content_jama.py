import hashlib
import os.path

import pandas as pd
from bs4 import BeautifulSoup

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

journal_lists = pd.read_csv('./jama_journal.v2.csv', sep='\t', header=None)


def get_title(soup):
    h1_title = soup.findAll('h1', {"class": "meta-article-title"})
    if h1_title:
        return h1_title[0].get_text()
    else:
        h1_title = soup.findAll('h1', {"class": "meta-article-title "})
        if h1_title:
            return h1_title[0].get_text()
        else:
            return None


def get_citation_views(soup):
    views_data = soup.findAll('span', {'class': "artmet-number sb-sc"})
    cite_data = soup.findAll('span', {'class': "artmet-number"})
    if len(cite_data) == 2:
        views, citation = int(views_data[0].get_text().replace(',', '')), int(cite_data[1].get_text().replace(',', ''))
    elif views_data:
        views, citation = int(views_data[0].get_text().replace(',', '')), 0
    else:
        views, citation = 0, 0
    return citation, views


def get_type_journal(soup):
    j_type = soup.findAll('div', {"class": "meta-article-type thm-col"})
    if j_type:
        return j_type[0].get_text()
    else:
        return None


def get_content(soup):
    content = soup.findAll('p', {'class': "para"})
    contents = ''
    for cont in content[:-2]:
        conts = cont.get_text()
        if '?' not in conts:
            contents += conts.replace('\n', ' ') + ' '
    return contents


# scrape each conditions.
journal_paper_content = []

import requests
#pip install requests-ip-rotator
from requests_ip_rotator import ApiGateway

# Create gateway object and initialise in AWS
gateway = ApiGateway("https://jamanetwork.com/")
gateway.start()

# Assign gateway to session
session = requests.Session()
session.mount("https://jamanetwork.com/", gateway)



def multi_process_scrap(i):
    try:
        r = session.get(i, headers=headers)
        print(i)
    except:
        print('ReTrying about 5 seconds')
        return None
    soup = BeautifulSoup(r.content)
    title = get_title(soup)

    # if title:
    name = hashlib.sha256(title.encode()).hexdigest()
    if not os.path.isfile("./jamav2/%s.csv" % name):
        citation, views = get_citation_views(soup)
        contents = get_content(soup)
        j_type = get_type_journal(soup)
        ddfs = pd.DataFrame([[title, contents, citation, views, j_type]],
                            columns=['title', 'contents', 'citation', 'views', 'j_type'])
        ddfs.to_csv("./jamav2/%s.csv" % name, sep='\t')
        return ddfs
    else:
        return None


# df_dask = ddf.from_pandas(journal_lists, npartitions=3)
# res = df_dask.map_partitions(lambda df: df.apply((lambda row: multi_process_scrap(*row)), axis=1)).compute(scheduler='multiprocessing')

# df_dask['output'] = df_dask.apply(lambda x: multi_process_scrap(x),axis=1).compute(scheduler='multiprocessing')
from multiprocessing import Pool

all_records = journal_lists[0].values

pool = Pool(processes=3)
results = pool.map(multi_process_scrap, all_records)
pool.close()
pool.join()

import pandas as pd
import os

csv_paths = os.listdir("jamav2/")
combine_df = pd.DataFrame()
for csv_path in csv_paths:
    # try:
    d = pd.read_csv(f'''./jamav2/{csv_path}''', sep='\t', index_col=0, lineterminator='\n')
    combine_df = combine_df.append(d, ignore_index=True)
    # except:
    #    print(csv_path)

combine_df.to_csv("./jama_journal_content.v2.csv", sep='\t',index=False)
#
elif_df = pd.read_csv('./elif_journal_content.v2.csv', sep='\t')
elif_df.dropna(subset=['content'],inplace=True)
elif_df.columns=['title','contents','citation','views']
elif_df['j_type']='elif'
combine_df = pd.read_csv('./jama_journal_content.v2.csv', sep='\t',lineterminator='\n',index_col=0)
final_df = pd.concat([combine_df, elif_df], ignore_index=True).reindex(combine_df.columns, axis='columns')

import uuid

final_df['id'] = [str(uuid.uuid4()) for _ in range(len(final_df.index))]
#
# final_df.to_csv("./all_journal_content.v2.csv", sep='\t',index=None)

final_df_2=pd.read_csv("./all_journal_content.csv",sep='\t',index_col=0,lineterminator='\n')

final_df=pd.concat([final_df, final_df_2], ignore_index=True).reindex(final_df.columns, axis='columns')

