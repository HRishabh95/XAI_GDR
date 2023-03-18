from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import pandas as pd
from requests_ip_rotator import ApiGateway

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# scrape each conditions.
journal_paper_links=[]
for i in range(1,500):
    print(i)
    base_url='https://jamanetwork.com/searchresults?f_FreeAccessFilter=true&f_ArticleTypeDisplayName=ResearchANDCase+ReportANDPatient+Information&exPrm_qqq=%7bDEFAULT_BOOST_FUNCTION%7d%22covid%22&sort=Newest&page='
    #base_url='https://jamanetwork.com/searchresults?f_FreeAccessFilter=true&f_ArticleTypeDisplayName=ResearchANDCase+ReportANDPatient+Information&exPrm_qqq=%7bDEFAULT_BOOST_FUNCTION%7d%22covid%22&page='
    url=f'''{base_url}{i}'''
    try:
        r = requests.get(url, headers=headers)
    except:
        time.sleep(6)
        continue
    soup = BeautifulSoup(r.content)

    journal_paper_lists=soup.findAll('a',{'class':'article--abstract'})
    for journal_paper_list in journal_paper_lists:
        journal_paper_links.append(f'''{journal_paper_list.get('href')}''')
    time.sleep(0.5)

journal_paper_link=pd.DataFrame(journal_paper_links)
journal_paper_link.to_csv('./jama_journal.v2.csv',sep='\t',index=False)


