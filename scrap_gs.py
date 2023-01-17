from bs4 import BeautifulSoup
import requests, lxml, os, json
def scrape_one_google_scholar_page():
    # https://requests.readthedocs.io/en/latest/user/quickstart/#custom-headers
    headers = {
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
# https://requests.readthedocs.io/en/latest/user/quickstart/#passing-parameters-in-urls
    params = {
        'q': 'Association of Vitamin D Levels, Race/Ethnicity, and Clinical Characteristics With COVID-19 Test Results',  # search query
        'hl': 'en'       # language of the search
    }
    html = requests.get('https://scholar.google.com/scholar', headers=headers, params=params).text
    soup = BeautifulSoup(html, 'lxml')
# JSON data will be collected here
    data = []
# Container where all needed data is located
    for result in soup.select('.gs_r.gs_or.gs_scl'):
        title = result.select_one('.gs_rt').text
        title_link = result.select_one('.gs_rt a')['href']
        publication_info = result.select_one('.gs_a').text
        snippet = result.select_one('.gs_rs').text
        cited_by = result.select_one('#gs_res_ccl_mid .gs_nph+ a')['href']

        data.append({
                'title': title,
                'title_link': title_link,
                'publication_info': publication_info,
                'snippet': snippet,
                'cited_by': f'https://scholar.google.com{cited_by}',
            })
    print(json.dumps(data, indent = 2, ensure_ascii = False))

from scholarly import scholarly

import requests as rq
from bs4 import BeautifulSoup as bs
from urllib.parse import urlencode

params = {
        'q': 'Association of Vitamin D Levels, Race/Ethnicity, and Clinical Characteristics With COVID-19 Test Results',  # search query
        'hl': 'en'       # language of the search
    }
headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0",
           'accept-language': 'en-US,en',
           'accept': 'text/html,application/xhtml+xml,application/xml',
           'Server': 'scholar',
           }


def google_scholar(query, n_pages, since_year):
    data = []
    encoded_query = urlencode({"q": params['q']})
    for start in range(0, n_pages * 10, 10):
        url = "https://scholar.google.com/scholar?as_ylo=%s&%s&hl=fr&start=%s" % (since_year, encoded_query, start)
        resp = rq.get(url, headers=headers)
        soup = bs(resp.content, "lxml")
        print(soup)
        main_div = soup.find_all('div', {'id': 'gs_res_ccl_mid'})[0]
        divs = main_div.find_all('div', {'class': 'gs_r gs_or gs_scl'})
        for div in divs:
            data_cid = div['data-cid']
            print(data_cid)
            title = div.find_all('h3', {'class': 'gs_rt'})[0].text
            infos = div.find_all('div', {'class': 'gs_a'})[0].text

            # APA citation
            url_cite = "https://scholar.google.com/scholar?q=info:%s:scholar.google.com/&output=cite&scirp=0&hl=fr" % (
                data_cid)
            resp2 = rq.get(url_cite, headers=headers)




headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
    'referer': f"https://scholar.google.com/scholar?hl={params['hl']}&q={params['q']}"
}

html = requests.get(f'https://scholar.google.com/scholar?output=cite&q=info:aK3xwslR5gYJ:scholar.google.com',
                    headers=headers)


queries = ['Association of Vitamin D Levels, Race/Ethnicity, and Clinical Characteristics With COVID-19 Test Results']
import requests
from bs4 import BeautifulSoup as bs

with requests.Session() as s:
    for query in queries:
        url = 'https://scholar.google.com/scholar?q=' + query + '&ie=UTF-8&oe=UTF-8&hl=en&btnG=Search'
        r = s.get(url)
        soup = bs(r.content, 'lxml') # or 'html.parser'
        title = soup.select_one('h3.gs_rt a').text if soup.select_one('h3.gs_rt a') is not None else 'No title'
        link = soup.select_one('h3.gs_rt a')['href'] if title != 'No title' else 'No link'
        citations = soup.select_one('a:contains("Cited by")').text if soup.select_one('a:contains("Cited by")') is not None else 'No citation count'
        print(title, link, citations)