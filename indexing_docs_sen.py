import pandas as pd
import numpy as np
from transformers import pipeline
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForTokenClassification
ner_model='d4data/biomedical-ner-all'
tokenizer = AutoTokenizer.from_pretrained(ner_model)
model_ner = AutoModelForTokenClassification.from_pretrained(ner_model)
pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
import string
PUNCTUATIONS = string.punctuation.replace(".",'')


def remove_punctuation(text):
  trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
  return text.translate(trans)

def remove_whitespaces(text):
    return " ".join(text.split())

def clean_en_text(text):
  """
  text
  """
  text = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", text)
  #text= re.sub(r'\d+', '',text)
  text = remove_punctuation(text)
  text = remove_whitespaces(text)
  return text.strip().lower()

def trec_generate(f):
  df = pd.DataFrame(f, columns=['docno', 'text'])
  return df



import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

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

from rank_bm25 import BM25Okapi,BM25Plus,BM25L

docs_dfs=pd.read_csv("./docs/gen_docs_top_100.csv",sep='\t')
#
# qids = np.unique(docs_dfs.qid.values)
# sens_all=[]
# for qid in qids:
#     docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
#     for ii, doc_rows in docs_tops.iterrows():
#         print(qid,ii)
#         query=doc_rows['query']
#         texts=doc_rows['text']
#         corpus=[]
#         for i,sen in enumerate(texts.split(".")):
#             sen_clean=clean_en_text(sen)
#             if len(sen_clean.split())>2 and  sen_clean not in corpus:
#                 corpus.append(sen_clean)
#
#         tokenized_corpus = [doc.split(" ") for doc in corpus]
#         bm25 = BM25Okapi(tokenized_corpus)
#         tokenized_query = query.split(" ")
#         topical_sens=bm25.get_top_n(tokenized_query, corpus, n=3)
#         sens=''
#         for topical_sen in topical_sens:
#             sens+=topical_sen+"\t 1,"
#         sens_all.append(sens)
# docs_dfs['top_sens']=sens_all



qids = np.unique(docs_dfs.qid.values)
sens_all=[]
for qid in qids:
    print(qid)
    docs_tops = docs_dfs.loc[docs_dfs['qid'] == qid].sort_values(by=['rank'])
    for ii, doc_rows in docs_tops.iterrows():
        query=doc_rows['query']
        texts=doc_rows['text']
        corpus=[]
        texts=split_into_sentences(texts)
        sens = ''
        if len(texts)>1:
            for i,sen in enumerate(texts):
                sen_clean=clean_en_text(sen)
                if len(sen_clean.split())>2 and  sen_clean not in corpus:
                    corpus.append(sen_clean)
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.split(" ")
            topical_sens=bm25.get_top_n(tokenized_query, corpus, n=5)
            scores=-np.sort(-bm25.get_scores(tokenized_query))
            for sen_idx,topical_sen in enumerate(topical_sens):
                query_sen_entity_name = get_entity_name(pipe(query))
                arti_sen_entity_name = get_entity_name(pipe(topical_sen))
                simi=1
                if query_sen_entity_name and arti_sen_entity_name:
                    if query_sen_entity_name == arti_sen_entity_name:
                        sens += topical_sen + "\t %s," % (float(simi))
                    elif SequenceMatcher(None, query_sen_entity_name, arti_sen_entity_name).ratio() > 0.5:
                        sens += topical_sen + "\t %s," % (float(simi) * SequenceMatcher(None, query_sen_entity_name, arti_sen_entity_name).ratio()*0.5)
                    else:
                        sens += topical_sen + "\t %s," % (float(simi) * 0.01)
                else:
                    sens += topical_sen + "\t %s," % (float(scores[sen_idx]) * 0.5)
        sens_all.append([doc_rows['qid'], doc_rows['docno'], sens])
sens=pd.DataFrame(sens_all,columns=['qid','docno','top_sens'])
dfs_final=pd.merge(docs_dfs,sens,on=['qid','docno'])

dfs_final.to_csv('./docs/docs_all_sens_bm25_ner.csv', index=None, sep=';')
