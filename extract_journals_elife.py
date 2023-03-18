from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# scrape each conditions.
journal_paper_links=[]
for i in range(1,653):
    print(i)
    base_url="https://elifesciences.org/search?for=&types%5B0%5D=research&sort=date&order=descending&page="
    url=f'''{base_url}{i}'''
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content)

    journal_paper_lists=soup.findAll('a',{'class':'teaser__header_text_link'})
    base_url_journal='https://elifesciences.org/'
    for journal_paper_list in journal_paper_lists:
        journal_paper_links.append(f'''{base_url_journal}{journal_paper_list.get('href')}''')

journal_paper_link=pd.DataFrame(journal_paper_links)
journal_paper_link.to_csv('./elif_journal.v2.csv',sep='\t',index=False)
