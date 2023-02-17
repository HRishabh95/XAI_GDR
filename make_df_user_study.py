from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI").to('cuda')

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch



import pandas as pd
import ast
import nltk

nltk.download('punkt')  # one time execution
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import difflib
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

ner_model = 'd4data/biomedical-ner-all'
tokenizer_ner = AutoTokenizer.from_pretrained(ner_model)
model_ner = AutoModelForTokenClassification.from_pretrained(ner_model)
pipe = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner,
                aggregation_strategy="simple")  # pass device=0 if using gpu


def get_entity_name(entities):
    if len(entities) > 0:
        return_word = []
        for entity in entities:
            if entity['entity_group'] == 'Medication':
                if entity['word'] not in return_word:
                    return_word.append(entity['word'])
        if len(return_word) > 0:
            return return_word
        else:
            return None
    else:
        return None


def get_nli(sentence):
    sent_1, sent_2 = sentence[0].split("\t")[0], sentence[0].split("\t")[1]
    text = f"mednli: sentence1: {sent_1} sentence2: {sent_2}"
    encoding = tokenizer.encode_plus(text, padding='max_length', max_length=256, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to('cuda'), encoding["attention_mask"].to('cuda')
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=8,
        early_stopping=True
    )
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return line

import errno
import os

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

import re
import string
PUNCTUATIONS = string.punctuation.replace('.','')


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

def utils_split_sentences(a, b):
    ## find clean matches
    match = difflib.SequenceMatcher(isjunk=None, a=a, b=b, autojunk=True)
    lst_match = [block for block in match.get_matching_blocks() if block.size > 20]

    ## difflib didn't find any match
    if len(lst_match) == 0:
        lst_a, lst_b = nltk.sent_tokenize(a), nltk.sent_tokenize(b)

    ## work with matches
    else:
        first_m, last_m = lst_match[0], lst_match[-1]

        ### a
        string = a[0: first_m.a]
        lst_a = [t for t in nltk.sent_tokenize(string)]
        for n in range(len(lst_match)):
            m = lst_match[n]
            string = a[m.a: m.a + m.size]
            lst_a.append(string)
            if n + 1 < len(lst_match):
                next_m = lst_match[n + 1]
                string = a[m.a + m.size: next_m.a]
                lst_a = lst_a + [t for t in nltk.sent_tokenize(string)]
            else:
                break
        string = a[last_m.a + last_m.size:]
        lst_a = lst_a + [t for t in nltk.sent_tokenize(string)]

        ### b
        string = b[0: first_m.b]
        lst_b = [t for t in nltk.sent_tokenize(string)]
        for n in range(len(lst_match)):
            m = lst_match[n]
            string = b[m.b: m.b + m.size]
            lst_b.append(string)
            if n + 1 < len(lst_match):
                next_m = lst_match[n + 1]
                string = b[m.b + m.size: next_m.b]
                lst_b = lst_b + [t for t in nltk.sent_tokenize(string)]
            else:
                break
        string = b[last_m.b + last_m.size:]
        lst_b = lst_b + [t for t in nltk.sent_tokenize(string)]

    return lst_a, lst_b


import string

PUNCTUATIONS = string.punctuation.replace('.', '')


def remove_punctuation(text):
    trans = str.maketrans(dict.fromkeys(PUNCTUATIONS, ' '))
    return text.translate(trans)


def remove_whitespaces(text):
    return " ".join(text.split())


def clean_en_text(text):
    """
    text
    """
    text = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", text)
    # text= re.sub(r'\d+', '',text)
    text = remove_punctuation(text)
    text = remove_whitespaces(text)
    return text.strip().lower()






user_study_df = pd.read_csv('./docs_user_study', sep=' ', names=['docno', 'qid'])
#simi_score = pd.read_csv('./experiments/dtop100_jtop10/gen_ner_func_manual_sens_similarity_score_sw_bm25.csv', sep='\t')
simi_score = pd.read_csv('./experiments/dtop100_jtop10/gen_ner_func_manual_sens_similarity_score_sw_biobert.csv',sep='\t')
journal_df=pd.read_csv('journal_citation.csv',sep='\t')

journal_cite_dict={}

for ii,jour_df in journal_df.iterrows():
    journal_cite_dict[jour_df['docno']]=jour_df['cite']

journal_url_dict={}

for ii,jour_df in journal_df.iterrows():
    journal_url_dict[jour_df['docno']]=jour_df['url']
user_study_df.dropna(inplace=True)

sentences_dicts = []
journal_dicts=[]
for ii, docs in user_study_df.iterrows():
    print(ii)
    docs_df = simi_score.loc[(simi_score['docno'] == docs['docno']) & (simi_score['qid'] == docs['qid'])]
    sentences_support = []
    journal_dicts=[]
    for ii, docs_top in docs_df.iterrows():
        sentens = ast.literal_eval(docs_top['scores'])
        for senten in sentens:
            if senten[1] > 0.0:
                if get_nli(senten) == 'entailment' or get_nli(senten) == 'neutral':
                    # doc_sen_entity_names = get_entity_name(pipe(senten[0][0].split("\t")[0]))
                    # journal_sen_entity_names = get_entity_name(pipe(senten[0][0].split("\t")[-1]))
                    # if doc_sen_entity_names and journal_sen_entity_names:
                    #     for doc_sen_entity_name in doc_sen_entity_names:
                    #         if doc_sen_entity_name in journal_sen_entity_names:
                    if senten not in sentences_support:
                        sentences_support.append(senten)
                        journal_dicts.append([senten,senten[1],docs_top['j_docno']])
    sentences_sup_sorted = sorted(journal_dicts, key=lambda t: t[1], reverse=True)
    sentence_dict = {}
    for sentences_sup_sort in sentences_sup_sorted:
        doc_sen = sentences_sup_sort[0][0].split("\t")[0]
        jou_sen = sentences_sup_sort[0][0].split("\t")[1]
        j_docno = sentences_sup_sort[-1]
        if sentences_sup_sort[1]>0.45:
            if doc_sen not in sentence_dict:
                sentence_dict[doc_sen] = []
                if j_docno in journal_cite_dict:
                    sentence_dict[doc_sen].append([jou_sen.rstrip().lstrip(),sentences_sup_sort[1],journal_cite_dict[j_docno],
                                                   journal_url_dict[j_docno]])
            else:
                if j_docno in journal_cite_dict:
                    sentence_dict[doc_sen].append([jou_sen.rstrip().lstrip(),sentences_sup_sort[1],journal_cite_dict[j_docno],
                                                   journal_url_dict[j_docno]])
    sentences_dicts.append(sentence_dict)

user_study_df['sentence_dicts'] = sentences_dicts
user_study_df.to_csv('docs_func_user_study_sen_j_id_055_bm25.csv', sep='\t', index=False)


def get_headline(abstract):
    model_name = "snrspeaks/t5-one-line-summary"
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer.encode("summarize: " + abstract, return_tensors="pt", add_special_tokens=True)
    generated_ids = model.generate(input_ids=input_ids,num_beams=5,max_length=100,repetition_penalty=2.5,length_penalty=1,early_stopping=True,num_return_sequences=1)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds

#model_name_sum = "google/pegasus-xsum"
# tokenizer_sum = PegasusTokenizer.from_pretrained(model_name_token)
# model_sum = PegasusForConditionalGeneration.from_pretrained(model_name_sum)

# def get_headline(doc_text):
#     src_text = [doc_text]
#     batch = tokenizer_sum(src_text, max_length=250,truncation=True, padding="longest", return_tensors="pt")
#     translated = model_sum.generate(**batch)
#     tgt_text = tokenizer_sum.batch_decode(translated, skip_special_tokens=True)
#     return tgt_text


import pandas as pd

user_study_df = pd.read_csv('docs_func_user_study_sen_j_id_055.csv', sep='\t')

import pyterrier as pt

if not pt.started():
    pt.init()
topics = pt.io.read_topics("/home/ubuntu/rupadhyay/dataset/TREC/topics.csv", format='singleline', tokenise=True)
topics['query'] = topics['query'].replace("can", '', regex=True)

user_study_df['qid'] = user_study_df['qid'].astype(int)
topics['qid'] = topics['qid'].astype(int)
user_study_qid_df = pd.merge(user_study_df, topics, on='qid')

user_study_qid_df.dropna(inplace=True)
import pickle


def trec_generate(f):
    df = pd.DataFrame(f, columns=['docno', 'text'])
    return df


wic_data = '/home/ubuntu/rupadhyay/dataset/TREC/trec_20_wic_top10_en_nd.csv'
f = pickle.load(open(wic_data, 'rb'))
df_docs = trec_generate(f)

text=df_docs[df_docs['docno']=='06413b57-7f8d-4278-a990-ff2ba9c78517']

user_study_qid_df = pd.merge(user_study_qid_df, df_docs, on='docno')

id=0
doc_coloured = []
for ii, user_study_qid_data in user_study_qid_df.iterrows():
    sents=''
    text_coloured=''
    texts=user_study_qid_data['text']
    header=get_headline(texts)
    import ast, re
    sens = user_study_qid_data['sentence_dicts']
    if not isinstance(sens, dict):
        sens = ast.literal_eval(sens)
    sens_a = list(sens.keys())
    if sens_a:
        #sens_a = sens_a[0]
        #for sen_a in sens_a:
        text_b = user_study_qid_data['text']
        lst_b = split_into_sentences(text_b)
        first_text = []
        for i in lst_b:
            i_clean = clean_en_text(i)
            for j in sens_a:
                j = clean_en_text(j)
                if i_clean == j:
                    #first_text.append('<span style="background-color:rgba(255,215,0,0.3);"> ' + i + ' </span>')
                    first_text.append('<b><i>'+i+'</i></b>')
                    text_coloured+=i+"\t"
                else:
                    first_text.append(i)
        first_text = ' '.join(first_text)
        if len(header[0])>2:
            doc_coloured.append([user_study_qid_data['qid'], user_study_qid_data['docno'], first_text,header[0],text_coloured])

doc_coloured=pd.DataFrame(doc_coloured,columns=['qid','docno','text_coloured','header','sentence'])
combined_df=pd.merge(user_study_qid_df,doc_coloured,on=['qid','docno'])
combined_df.drop_duplicates(subset=['header'],inplace=True)

im_sentences = []
for ii, combined in combined_df.iterrows():
    if not isinstance(combined['sentence_dicts'], dict):
        im_sentences.append(list(ast.literal_eval(combined['sentence_dicts']).items())[0][0])
    else:
        im_sentences.append(list(combined['sentence_dicts'].items())[0][0])

combined_df['im_sent']=im_sentences
combined_df.to_csv('docs_func_user_study_xai_j_id_bm25_multiple_sens.v1.csv',sep='\t',index=False)


