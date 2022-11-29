#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Library
import re
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle


# In[2]:


factory = StopWordRemoverFactory()
sastrawi_stopword = factory.get_stop_words()

# create path url for each stopword
path_stopwords = []

# combine stopwords
stopwords_l = sastrawi_stopword
for path in path_stopwords:
    response = requests.get(path)
    stopwords_l += response.text.split('\n')

# create dictionary with unique stopword
st_words = set(stopwords_l)

# result stopwords
stop_words = st_words


# In[3]:


# Function Preprocessing

def case_folding(text):
    text = text.lower()  # lowercase
    return text


def emoji(text):
    text = re.sub(r'[^\x00-\x7f]', r'', text)  # Remove non ASCII chars
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    return text


def cleaning_text(text):
    # Cleaning text
    text = re.sub(r'@[\w]*', ' ', text)  # Remove mention handle user (@)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r'\\u\w\w\w\w', '', text)  # Remove link web
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#([^\s]+)', '', text)  # Remove #tagger
    # Remove simbol, angka dan karakter aneh
    text = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", text)
    return text


def replaceThreeOrMore(text):
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gol).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1", text)


def tokenize(text):
    return word_tokenize(text)


def convertToSlangword(text):
    # Membuka dictionary slangword
    kamus_slangword = eval(open("kamus.txt").read())
    # Search pola kata (contoh kpn -> kapan)
    pattern = re.compile(r'\b( ' + '|'.join(kamus_slangword.keys())+r')\b')
    content = []
    for kata in text:
        # Replace slangword berdasarkan pola review yg telah ditentukan
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()], kata)
        content.append(filteredSlang.lower())
    text = content
    return text


def ganti_negasi(w):
    # print(w)
    kamus_negasi = eval(open("negasi.txt").read())  # Membuka dictionary negasi
    w_splited = w.split(' ')
    # print(w_splited)
    if 'tidak' in w_splited:
        index_negasi = w_splited.index('tidak')
        #  print(index_negasi)
        for i, k in enumerate(w_splited):
            if k in kamus_negasi and w_splited[i-1] == 'tidak':
                w_splited[i] = kamus_negasi[k]
                #  print(w_splited)
    return ' '.join(w_splited)


def remove_stopword(text, stop_words=stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)


def stemming_and_lemmatization(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# In[4]:


# Skema Preprocessing

def prepro_1(data):
    data['Clean_Twt'] = data['text'].apply(case_folding)
    data['Clean_Twt'] = data['Clean_Twt'].apply(emoji)
    data['Clean_Twt'] = data['Clean_Twt'].astype(str)
    data['Stopwords'] = data['Clean_Twt'].apply(remove_stopword)
    data['Stem'] = data['Stopwords'].apply(stemming_and_lemmatization)
    data['tweet'] = data['Stem']
    data = data[['text', 'sentiment']]
    return data


def prepro_2(data):
    #data['text'] = data['text'].astype(str)
    data['Clean_Twt'] = data['text'].apply(case_folding)
    data['Clean_Twt'] = data['Clean_Twt'].apply(emoji)
    data['Clean_Twt'] = data['Clean_Twt'].apply(cleaning_text)
    data['Clean_Twt'] = data['Clean_Twt'].astype(str)
    data['Repeat'] = data['Clean_Twt'].apply(replaceThreeOrMore)
    data['Tokenize_Tweet'] = data['Repeat'].apply(tokenize)
    data['Slang_Tweet'] = data['Tokenize_Tweet'].apply(convertToSlangword)
    data['Slang_Tweet'] = data['Slang_Tweet'].apply(" ".join)
    data['Negasi'] = data['Slang_Tweet'].apply(ganti_negasi)
    data['Stopwords'] = data['Negasi'].apply(remove_stopword)
    data['Stem'] = data['Stopwords'].apply(stemming_and_lemmatization)
    #data['tweet'] = data['Stopwords'].apply(tokenize)
    data['text'] = data['Stem']
    data = data[['text', 'sentiment']]
    return data


# In[ ]:
