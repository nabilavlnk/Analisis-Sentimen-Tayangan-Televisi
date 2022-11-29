from cgitb import text
from ntpath import join
from os import remove
from tkinter.ttk import Scale
from flask import Flask, render_template, jsonify, request
import csv

import pandas as pd
import preprocessing
import re
import nltk
import csv
import matplotlib.pyplot as plt
import sklearn.svm
import ast

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle

import sys
import json
import base64

app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():  # fungsi yang akan dijalankan ketike route dipanggil
    return render_template('index.html')


@app.route("/dataset", methods=['GET', 'POST'])
def dataset():
    with open('data.csv', encoding='latin-1') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
            if not first_line:
                dataset.append({
                    "text": row[0],
                    "sentiment": row[1]
                })
            else:
                first_line = False
    return render_template('dataset.html', menu='dataset', submenu='data', dataset=dataset)


@app.route("/cek_sentimen")
def cek_sentimen():
    return render_template('cek_sentimen.html')


@app.route("/cek_sentimen_nb")
def cek_sentimen_nb():
    return render_template('cek_sentimen_nb.html')


@app.route("/hasil_uji", methods=["GET"])
def hasil_uji():
    # load model
    load_TFIDF_all = pickle.load(open("tfidf_2_full.pkl", 'rb'))
    loaded_SVM_all = pickle.load(open("SVM2_full.pkl", 'rb'))

    load_TFIDF1 = pickle.load(open("tfidf_2_full.pkl", 'rb'))
    loaded_SVM1 = pickle.load(open("SVM2_full.pkl", 'rb'))

    user_data = request.args.get("sub")
    subject = user_data
    user_data = str(user_data)

    subject = request.args.get("sub")
    subject = [subject]

    result_prepro = {}
    result = {}

    def case_folding(tokens):
        return tokens.lower()

    test_casefolding = []
    for i in range(0, len(subject)):
        test_casefolding.append(case_folding(subject[i]))

    result['casefolding'] = ' '.join(
        list(map(lambda x: str(x), test_casefolding)))
    casefolding = result['casefolding']

    import ast

    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])

    # Skenario I (Cleaning + Stopwords + Stemming)
    result_prepro['teks'] = ' '.join(
        list(map(lambda x: str(x), test_casefolding)))
    casefolding_2 = result_prepro['teks']
    result_prepro['emoji'] = preprocessing.emoji(
        result_prepro['teks'])
    emoji_2 = result_prepro['emoji']
    result_prepro['stopword'] = preprocessing.remove_stopword(
        result_prepro['emoji'])
    stopword_2 = result_prepro['stopword']
    result_prepro['stem'] = preprocessing.stemming_and_lemmatization(
        result_prepro['stopword'])
    stem_2 = result_prepro['stem']
    result_prepro['token'] = preprocessing.tokenize(
        result_prepro['stem'])
    text_final_prepro = result_prepro['token']

    # TF-IDF Skenario I
    result_prepro = stem_2
    vect_prepro = load_TFIDF1.transform([result_prepro]).toarray()

    # Support Vector Machine Skenario I
    prediksisvm_prepro = loaded_SVM1.predict(vect_prepro)
    proba_svm_prepro = loaded_SVM1.predict_proba(vect_prepro)[0]
    probabilitas_svm_prepro = proba_svm_prepro

    probabilitas_svm_new_prepro = "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas_svm_prepro[0], probabilitas_svm_prepro[-1], probabilitas_svm_prepro[1])
    probabilitas_svm_prepro = probabilitas_svm_new_prepro

    # Output kelas Skenario I
    if prediksisvm_prepro == 1:
        hasil_kelas_prepro = 'Positif'
    elif prediksisvm_prepro == (-1):
        hasil_kelas_prepro = 'Negatif'
    elif prediksisvm_prepro == 0:
        hasil_kelas_prepro = 'Netral'

    # Skenario II (all preprocessing)
    result['teks'] = ' '.join(
        list(map(lambda x: str(x), test_casefolding)))
    casefolding = result['teks']
    result['emoji'] = preprocessing.emoji(
        result['teks'])
    emoji = result['emoji']
    result['cleaning'] = preprocessing.cleaning_text(
        result['emoji'])
    cleaning = result['cleaning']
    result['repetition'] = preprocessing.replaceThreeOrMore(
        result['cleaning'])
    repetition = result['repetition']
    result['tokenize'] = preprocessing.tokenize(
        result['repetition'])
    tokenize = result['tokenize']
    result['slangword'] = preprocessing.convertToSlangword(
        result['tokenize'])
    result['slangword'] = ' '.join(
        list(map(lambda x: str(x), result['slangword'])))
    slangword = result['slangword']
    result['negasi'] = preprocessing.ganti_negasi(
        result['slangword'])
    negasi = result['negasi']
    result['stopword'] = preprocessing.remove_stopword(
        result['negasi'])
    stopword = result['stopword']
    result['stem'] = preprocessing.stemming_and_lemmatization(
        result['stopword'])
    stem = result['stem']
    result['token'] = preprocessing.tokenize(result['stem'])

    text_final = result['token']

    # TF-IDF
    result_ = stem
    vect = load_TFIDF_all.transform([result_]).toarray()

    prediksisvm = loaded_SVM_all.predict(vect)[0]
    proba_svm = loaded_SVM_all.predict_proba(vect)[0]
    probabilitas_svm = proba_svm

    probabilitas_svm_new = "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas_svm[0], probabilitas_svm[-1], probabilitas_svm[1])
    probabilitas_svm = probabilitas_svm_new

    # Output kelas Skenario II
    if prediksisvm == 1:
        hasil_kelas = 'Positif'
    elif prediksisvm == (-1):
        hasil_kelas = 'Negatif'
    elif prediksisvm == 0:
        hasil_kelas = 'Netral'

    return render_template('hasil_uji.html',
                           subject=subject,
                           casefolding=casefolding,
                           casefolding_2=casefolding_2,
                           cleaning=cleaning,
                           emoji=emoji,
                           emoji_2=emoji_2,
                           repetition=repetition,
                           tokenize=tokenize,
                           slangword=slangword,
                           negasi=negasi,
                           stopword=stopword,
                           stopword_2=stopword_2,
                           stem=stem,
                           stem_2=stem_2,
                           text_final_prepro=text_final_prepro,
                           text_final=text_final,
                           user_data=user_data,
                           probabilitas_svm=probabilitas_svm,
                           probabilitas_svm_prepro=probabilitas_svm_prepro,
                           hasil_kelas=hasil_kelas,
                           hasil_kelas_prepro=hasil_kelas_prepro)


@app.route("/hasil_uji_nb", methods=["GET"])
def hasil_uji_nb():
    # load model
    load_TFIDF_all = pickle.load(open("tfidf_2_full.pkl", 'rb'))
    loaded_NB_all = pickle.load(open("NB2_full.pkl", 'rb'))

    load_TFIDF1 = pickle.load(open("tfidf_2_full.pkl", 'rb'))
    loaded_NB1 = pickle.load(open("NB2_full.pkl", 'rb'))

    # input data
    user_data = request.args.get("sub")
    subject = user_data
    user_data = str(user_data)

    subject = [subject]

    result_1 = {}
    result = {}

    def join_text_list(texts):
        texts = ast.literal_eval(texts)
        return ' '.join([text for text in texts])

    def case_folding(tokens):
        return tokens.lower()

    test_casefolding = []
    for i in range(0, len(subject)):
        test_casefolding.append(case_folding(subject[i]))

    # Skenario I (Cleaning + Stopwords + Stemming)
    result_1['teks'] = ' '.join(
        list(map(lambda x: str(x), test_casefolding)))
    result_1['emoji'] = preprocessing.emoji(
        result_1['teks'])
    emoji_2 = result_1['emoji']
    result_1['stopword'] = preprocessing.remove_stopword(
        result_1['emoji'])
    stopword_2 = result_1['stopword']
    result_1['stem'] = preprocessing.stemming_and_lemmatization(
        result_1['stopword'])
    stem_2 = result_1['stem']
    result_1['token'] = preprocessing.tokenize(
        result_1['stem'])
    text_prepro = result_1['token']

    # TF-IDF Skenario I
    result_prepro = stem_2
    vect_prepro = load_TFIDF1.transform([result_prepro]).toarray()

    # Naive Bayes Skenario I
    prediksinb_prepro = loaded_NB1.predict(vect_prepro)
    proba_nb_prepro = loaded_NB1.predict_proba(vect_prepro)[0]
    probabilitas_nb_prepro = proba_nb_prepro

    probabilitas_nb_new_prepro = "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas_nb_prepro[0], probabilitas_nb_prepro[-1], probabilitas_nb_prepro[1])
    probabilitas_nb_prepro = probabilitas_nb_new_prepro

    # Output kelas Skenario I
    if prediksinb_prepro == 1:
        hasil_kelas_prepro = 'Positif'
    elif prediksinb_prepro == (-1):
        hasil_kelas_prepro = 'Negatif'
    elif prediksinb_prepro == 0:
        hasil_kelas_prepro = 'Netral'

    # Skenario II (All preprocessing)
    result['casefolding'] = ' '.join(
        list(map(lambda x: str(x), test_casefolding)))
    casefolding = result['casefolding']
    result['emoji'] = preprocessing.emoji(
        result['casefolding'])
    emoji = result['emoji']
    result['cleaning'] = preprocessing.cleaning_text(
        result['emoji'])
    cleaning = result['cleaning']
    result['repetition'] = preprocessing.replaceThreeOrMore(
        result['cleaning'])
    repetition = result['repetition']
    result['tokenize'] = preprocessing.tokenize(
        result['repetition'])
    tokenize = result['tokenize']
    result['slangword'] = preprocessing.convertToSlangword(
        result['tokenize'])
    result['slangword'] = ' '.join(
        list(map(lambda x: str(x), result['slangword'])))
    slangword = result['slangword']
    result['negasi'] = preprocessing.ganti_negasi(
        result['slangword'])
    negasi = result['negasi']
    result['stopword'] = preprocessing.remove_stopword(
        result['negasi'])
    stopword = result['stopword']
    result['stem'] = preprocessing.stemming_and_lemmatization(
        result['stopword'])
    stem = result['stem']
    result['token'] = preprocessing.tokenize(result['stem'])

    text_final = result['token']

    # TF-IDF Skenario III
    result_ = stem
    vect = load_TFIDF_all.transform([result_]).toarray()

    # Naive Bayes Skenario III
    prediksinb = loaded_NB_all.predict(vect)
    proba_nb = loaded_NB_all.predict_proba(vect)[0]
    probabilitas_nb = proba_nb

    probabilitas_nb_new = "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas_nb[0], probabilitas_nb[-1], probabilitas_nb[1])
    probabilitas_nb = probabilitas_nb_new

    # Output kelas Skenario II
    if prediksinb == 1:
        hasil_kelas = 'Positif'
    elif prediksinb == (-1):
        hasil_kelas = 'Negatif'
    elif prediksinb == 0:
        hasil_kelas = 'Netral'

    if vect is not None:
        try:
            return render_template('hasil_uji_nb.html',
                                   subject=subject,
                                   casefolding=casefolding,
                                   cleaning=cleaning,
                                   emoji=emoji,
                                   emoji_2=emoji_2,
                                   repetition=repetition,
                                   tokenize=tokenize,
                                   slangword=slangword,
                                   negasi=negasi,
                                   stopword=stopword,
                                   stem=stem,
                                   text_prepro=text_prepro,
                                   text_final=text_final,
                                   stopword_2=stopword_2,
                                   stem_2=stem_2,
                                   user_data=user_data,
                                   probabilitas_nb=probabilitas_nb,
                                   probabilitas_nb_prepro=probabilitas_nb_prepro,
                                   hasil_kelas=hasil_kelas,
                                   hasil_kelas_prepro=hasil_kelas_prepro)
        except Exception as e:
            print(f"kesalahan: {e}")
    else:
        return render_template('hasil_uji_nb.html', subject=subject)


@app.route("/statistik")
def statistik():
    return render_template('statistik.html')


if __name__ == "__main__":
    app.run(debug=True)
